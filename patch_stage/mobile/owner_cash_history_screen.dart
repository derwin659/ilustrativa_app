import 'package:flutter/material.dart';
import '../../constants/admin_permission_keys.dart';
import '../../model/owner_cash_models.dart';
import '../../model/owner_sale_catalog_models.dart';
import '../../services/owner_cash_service.dart';
import '../../services/owner_sale_catalog_service.dart';
import '../../services/session_service.dart';
import '../../services/tenant_money_service.dart';
import '../../widgets/electronic_document_card.dart';
import '../../widgets/super_gods_ui.dart';
import 'business_labels.dart';
import 'owner_cash_movement_sheet.dart';

class OwnerCashHistoryScreen extends StatefulWidget {
  final int branchId;
  final String branchName;

  const OwnerCashHistoryScreen({
    super.key,
    required this.branchId,
    required this.branchName,
  });

  @override
  State<OwnerCashHistoryScreen> createState() => _OwnerCashHistoryScreenState();
}

class _OwnerCashHistoryScreenState extends State<OwnerCashHistoryScreen> {
  final OwnerCashService _service = OwnerCashService();
  final OwnerSaleCatalogService _catalogService = OwnerSaleCatalogService();
  BusinessLabels _ownerLabels = BusinessLabels.fromType('BARBERSHOP');

  bool _loading = true;
  bool _canManageFund = false;
  String? _error;
  List<CashRegisterModel> _items = [];
  List<CashFundMovementModel> _fundMovements = [];
  CashFundRangeSummaryModel? _fundSummary;

  late DateTime _from;
  late DateTime _to;

  @override
  void initState() {
    super.initState();
    _loadBusinessLabels();
    final now = DateTime.now();
    _to = now;
    _from = now.subtract(const Duration(days: 30));
    _load();
  }

  String _money(double value) => TenantMoneyService.format(value);

  Future<void> _loadBusinessLabels() async {
    final businessType = await SessionService.getBusinessType();
    if (!mounted) return;
    setState(() {
      _ownerLabels = BusinessLabels.fromType(businessType);
    });
  }

  String _normalizePaymentCode(String value) {
    return value
        .trim()
        .toUpperCase()
        .replaceAll('Á', 'A')
        .replaceAll('É', 'E')
        .replaceAll('Í', 'I')
        .replaceAll('Ó', 'O')
        .replaceAll('Ú', 'U')
        .replaceAll('EFECTIVO', 'CASH')
        .replaceAll('TARJETA', 'CARD')
        .replaceAll('TRANSFERENCIA', 'TRANSFER');
  }

  Map<String, double> _rangePaymentsSummary() {
    final result = <String, double>{};

    for (final cash in _items) {
      for (final p in cash.paymentMethodsSummary) {
        final code = _normalizePaymentCode(p.paymentMethod);
        if (code.isEmpty || p.totalAmount <= 0) continue;
        result[code] = (result[code] ?? 0) + p.totalAmount;
      }
    }

    return result;
  }

  Map<String, double> _rangePaymentBalances() {
    final result = <String, double>{};

    for (final cash in _items) {
      final source = cash.paymentMethodBalances.isNotEmpty
          ? cash.paymentMethodBalances
          : cash.paymentMethodsSummary;

      for (final p in source) {
        final code = _normalizePaymentCode(p.paymentMethod);
        if (code.isEmpty || code == 'FREE') continue;
        result[code] = (result[code] ?? 0) + p.totalAmount;
      }
    }

    return result;
  }

  double get _rangeSalesTotal =>
      _items.fold<double>(0, (sum, cash) => sum + cash.salesTotal);

  double get _rangeIncomeTotal =>
      _items.fold<double>(0, (sum, cash) => sum + cash.movementsIncome);

  double get _rangeExpenseTotal => _items.fold<double>(
        0,
        (sum, cash) =>
            sum +
            cash.movementsExpenseGeneral +
            cash.movementsAdvanceBarber +
            cash.movementsPaymentBarber,
      );

  double get _rangeNetTotal =>
      _rangeSalesTotal + _rangeIncomeTotal - _rangeExpenseTotal;

  double get _rangeCashExpectedTotal => _items.fold<double>(
        0,
        (sum, cash) => sum + cash.closingAmountExpected,
      );

  double get _rangeFundIncome => _fundMovements
      .where((movement) => movement.isIncome)
      .fold<double>(0, (sum, movement) => sum + movement.amount);

  double get _rangeFundExpense => _fundMovements
      .where((movement) => !movement.isIncome)
      .fold<double>(0, (sum, movement) => sum + movement.amount);

  double get _rangeFundNet => _fundMovements.fold<double>(
      0, (sum, movement) => sum + movement.signedAmount);

  double get _rangeCashExpenseTotal =>
      _items.fold<double>(0, (sum, cash) => sum + cash.cashMovementsExpense);

  double get _rangeOtherExpenseTotal =>
      (_rangeExpenseTotal - _rangeCashExpenseTotal)
          .clamp(0, double.infinity)
          .toDouble();

  String _dateText(String? raw) {
    if (raw == null || raw.trim().isEmpty) return 'Fecha no disponible';

    final date = DateTime.tryParse(raw);
    if (date == null) return raw;

    final dd = date.day.toString().padLeft(2, '0');
    final mm = date.month.toString().padLeft(2, '0');
    final yyyy = date.year.toString();
    final hh = date.hour.toString().padLeft(2, '0');
    final min = date.minute.toString().padLeft(2, '0');

    return '$dd/$mm/$yyyy · $hh:$min';
  }

  String _dateOnlyText(DateTime date) {
    final dd = date.day.toString().padLeft(2, '0');
    final mm = date.month.toString().padLeft(2, '0');
    final yyyy = date.year.toString();
    return '$dd/$mm/$yyyy';
  }

  String _statusLabel(String status) {
    switch (status.toUpperCase()) {
      case 'OPEN':
        return 'Abierta';
      case 'CLOSED':
        return 'Cerrada';
      case 'AUTO_CLOSED':
        return 'Cierre automático';
      default:
        return status;
    }
  }

  Color _statusColor(String status) {
    switch (status.toUpperCase()) {
      case 'OPEN':
        return const Color(0xFF16A34A);
      case 'CLOSED':
        return const Color(0xFF2563EB);
      case 'AUTO_CLOSED':
        return const Color(0xFFC2410C);
      default:
        return const Color(0xFF667085);
    }
  }

  Future<void> _load() async {
    if (!mounted) return;

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final canManageFund = await SessionService.hasPermission(
        AdminPermissionKeys.cashFundManage,
      );
      final results = await Future.wait<Object>([
        _service.getCashHistory(
          branchId: widget.branchId,
          from: _from,
          to: _to,
        ),
        if (canManageFund)
          _service.getCashFundMovements(
            branchId: widget.branchId,
            from: _from,
            to: _to,
          ),
        if (canManageFund)
          _service.getCashFundMovementSummary(
            branchId: widget.branchId,
            from: _from,
            to: _to,
          ),
      ]);

      if (!mounted) return;

      setState(() {
        _canManageFund = canManageFund;
        _items = results.first as List<CashRegisterModel>;
        _fundMovements = canManageFund
            ? results[1] as List<CashFundMovementModel>
            : const [];
        _fundSummary =
            canManageFund ? results[2] as CashFundRangeSummaryModel : null;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _error = SuperGodsUi.humanError(e);
        _loading = false;
      });
    }
  }

  bool _editableFund(CashFundMovementModel item) =>
      const {
        'MANUAL_DEPOSIT',
        'MANUAL_WITHDRAWAL',
        'ADJUSTMENT_IN',
        'ADJUSTMENT_OUT'
      }.contains(item.type.toUpperCase()) &&
      item.cashRegisterId == null;

  Future<String?> _auditReason(String title) async {
    final controller = TextEditingController();
    final value = await showDialog<String>(
        context: context,
        builder: (dialogContext) => AlertDialog(
                title: Text(title),
                content: TextField(
                    controller: controller,
                    autofocus: true,
                    maxLines: 3,
                    decoration: const InputDecoration(
                        labelText: 'Motivo de auditoría',
                        hintText: 'Explica por qué realizas esta corrección',
                        border: OutlineInputBorder())),
                actions: [
                  TextButton(
                      onPressed: () => Navigator.pop(dialogContext),
                      child: const Text('Cancelar')),
                  FilledButton(
                      onPressed: () {
                        final text = controller.text.trim();
                        if (text.isNotEmpty) Navigator.pop(dialogContext, text);
                      },
                      child: const Text('Continuar'))
                ]));
    controller.dispose();
    return value;
  }

  Future<void> _editFund(CashFundMovementModel item) async {
    if (!_editableFund(item)) return;
    final amount = TextEditingController(text: item.amount.toStringAsFixed(2));
    final concept = TextEditingController(text: item.concept);
    final note = TextEditingController(text: item.note ?? '');
    final accepted = await showDialog<bool>(
        context: context,
        builder: (dialogContext) => AlertDialog(
                title: const Text('Editar movimiento del fondo'),
                content: SingleChildScrollView(
                    child: Column(mainAxisSize: MainAxisSize.min, children: [
                  TextField(
                      controller: amount,
                      keyboardType:
                          const TextInputType.numberWithOptions(decimal: true),
                      decoration: const InputDecoration(labelText: 'Monto')),
                  TextField(
                      controller: concept,
                      decoration: const InputDecoration(labelText: 'Concepto')),
                  TextField(
                      controller: note,
                      maxLines: 2,
                      decoration: const InputDecoration(labelText: 'Detalle'))
                ])),
                actions: [
                  TextButton(
                      onPressed: () => Navigator.pop(dialogContext, false),
                      child: const Text('Cancelar')),
                  FilledButton(
                      onPressed: () => Navigator.pop(dialogContext, true),
                      child: const Text('Guardar'))
                ]));
    if (accepted == true && mounted) {
      final parsed = double.tryParse(amount.text.replaceAll(',', '.'));
      if (parsed == null || parsed <= 0 || concept.text.trim().isEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
            content: Text('Completa un monto válido y el concepto.')));
      } else {
        final reason = await _auditReason('Confirma la corrección');
        if (reason != null && mounted) {
          try {
            await _service.updateCashFundMovement(
                branchId: widget.branchId,
                movement: item,
                amount: parsed,
                concept: concept.text.trim(),
                note: note.text.trim().isEmpty ? null : note.text.trim(),
                auditReason: reason);
            await _load();
          } catch (e) {
            if (mounted)
              ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text(SuperGodsUi.humanError(e))));
          }
        }
      }
    }
    amount.dispose();
    concept.dispose();
    note.dispose();
  }

  Future<void> _deleteFund(CashFundMovementModel item) async {
    if (!_editableFund(item)) return;
    final reason = await _auditReason('Eliminar movimiento del fondo');
    if (reason == null || !mounted) return;
    final confirmed = await showDialog<bool>(
        context: context,
        builder: (dialogContext) => AlertDialog(
                title: const Text('¿Eliminar movimiento?'),
                content: Text(
                    'Se eliminará “${item.concept}”. El motivo quedará auditado.'),
                actions: [
                  TextButton(
                      onPressed: () => Navigator.pop(dialogContext, false),
                      child: const Text('Cancelar')),
                  FilledButton(
                      onPressed: () => Navigator.pop(dialogContext, true),
                      child: const Text('Eliminar'))
                ]));
    if (confirmed == true) {
      try {
        await _service.deleteCashFundMovement(
            branchId: widget.branchId,
            movementId: item.id,
            auditReason: reason);
        await _load();
      } catch (e) {
        if (mounted)
          ScaffoldMessenger.of(context)
              .showSnackBar(SnackBar(content: Text(SuperGodsUi.humanError(e))));
      }
    }
  }

  Future<void> _pickFrom() async {
    final picked = await _showPremiumDatePicker(
      initialDate: _from,
      helpText: 'Selecciona fecha inicial',
    );

    if (picked == null) return;

    setState(() {
      _from = picked;
      if (_from.isAfter(_to)) {
        _to = _from;
      }
    });

    await _load();
  }

  Future<void> _pickTo() async {
    final picked = await _showPremiumDatePicker(
      initialDate: _to,
      helpText: 'Selecciona fecha final',
    );

    if (picked == null) return;

    setState(() {
      _to = picked;
      if (_to.isBefore(_from)) {
        _from = _to;
      }
    });

    await _load();
  }

  Future<DateTime?> _showPremiumDatePicker({
    required DateTime initialDate,
    required String helpText,
  }) {
    const dark = Color(0xFF17131E);
    const gold = Color(0xFFD4AF37);
    const bg = Color(0xFFFFFCF7);

    return showDatePicker(
      context: context,
      initialDate: initialDate,
      firstDate: DateTime(2023),
      lastDate: DateTime.now().add(const Duration(days: 1)),
      helpText: helpText,
      cancelText: 'Cancelar',
      confirmText: 'Aceptar',
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            useMaterial3: true,
            colorScheme: const ColorScheme.light(
              primary: gold,
              onPrimary: dark,
              secondary: gold,
              onSecondary: dark,
              surface: bg,
              onSurface: dark,
              surfaceContainerHighest: Color(0xFFF6F0E3),
              onSurfaceVariant: Color(0xFF6B5B2A),
              outline: Color(0xFFEBDFAF),
            ),
            dialogTheme: DialogThemeData(
              backgroundColor: bg,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(28),
              ),
            ),
            datePickerTheme: DatePickerThemeData(
              backgroundColor: bg,
              surfaceTintColor: Colors.transparent,
              elevation: 10,
              shadowColor: Colors.black.withOpacity(0.10),
              headerBackgroundColor: bg,
              headerForegroundColor: dark,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(28),
              ),
              dayForegroundColor: WidgetStateProperty.resolveWith((states) {
                if (states.contains(WidgetState.selected)) return dark;
                if (states.contains(WidgetState.disabled)) {
                  return const Color(0xFFB7B0BC);
                }
                return dark;
              }),
              dayBackgroundColor: WidgetStateProperty.resolveWith((states) {
                if (states.contains(WidgetState.selected)) return gold;
                if (states.contains(WidgetState)) {
                  return const Color(0xFFFFF7D6);
                }
                return Colors.transparent;
              }),
              dayOverlayColor: WidgetStateProperty.all(
                gold.withOpacity(0.12),
              ),
              todayForegroundColor: WidgetStateProperty.all(dark),
              todayBackgroundColor: WidgetStateProperty.all(
                const Color(0xFFFFF7D6),
              ),
              todayBorder: const BorderSide(
                color: gold,
                width: 1.4,
              ),
              weekdayStyle: const TextStyle(
                color: Color(0xFF7A5A22),
                fontWeight: FontWeight.w900,
                fontSize: 12,
              ),
              dayStyle: const TextStyle(
                fontWeight: FontWeight.w800,
                fontSize: 13,
              ),
              yearStyle: const TextStyle(
                fontWeight: FontWeight.w800,
              ),
              cancelButtonStyle: TextButton.styleFrom(
                foregroundColor: const Color(0xFF7A7282),
                textStyle: const TextStyle(
                  fontWeight: FontWeight.w900,
                ),
              ),
              confirmButtonStyle: TextButton.styleFrom(
                foregroundColor: dark,
                textStyle: const TextStyle(
                  fontWeight: FontWeight.w900,
                ),
              ),
            ),
            textButtonTheme: TextButtonThemeData(
              style: TextButton.styleFrom(
                foregroundColor: dark,
                textStyle: const TextStyle(
                  fontWeight: FontWeight.w900,
                ),
              ),
            ),
          ),
          child: child!,
        );
      },
    );
  }

  void _showCashDetail(CashRegisterModel cash) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _CashHistoryDetailSheet(
        cash: cash,
        branchId: widget.branchId,
        from: _from,
        to: _to,
        money: _money,
        dateText: _dateText,
        statusLabel: _statusLabel,
        statusColor: _statusColor,
        canManageFund: _canManageFund,
      ),
    );
  }

  Future<void> _showAuditActivity() async {
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _CashAuditActivitySheet(
        branchId: widget.branchId,
        branchName: widget.branchName,
        from: _from,
        to: _to,
        dateText: _dateText,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    const bg = Color(0xFFF6F3ED);

    return Scaffold(
      backgroundColor: bg,
      appBar: AppBar(
        title: const Text('Historial de caja'),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black87,
        elevation: 0.6,
      ),
      body: RefreshIndicator(
        onRefresh: _load,
        child: ListView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(16),
          children: [
            _HeaderCard(
              branchName: widget.branchName,
              from: _dateOnlyText(_from),
              to: _dateOnlyText(_to),
              onFrom: _pickFrom,
              onTo: _pickTo,
              onActivity: _showAuditActivity,
            ),
            if (!_loading && _error == null && _items.isNotEmpty) ...[
              const SizedBox(height: 12),
              _RangeCashSummaryCard(
                payments: _rangePaymentBalances(),
                salesTotal: _rangeSalesTotal,
                incomeTotal: _rangeIncomeTotal,
                expenseTotal: _rangeExpenseTotal,
                cashExpenseTotal: _rangeCashExpenseTotal,
                otherExpenseTotal: _rangeOtherExpenseTotal,
                netTotal: _rangeNetTotal,
                cashExpectedTotal: _rangeCashExpectedTotal,
                money: _money,
              ),
            ],
            if (!_loading && _error == null && _canManageFund) ...[
              const SizedBox(height: 16),
              _FundMovementHistoryCard(
                movements: _fundMovements,
                totalIn: _rangeFundIncome,
                totalOut: _rangeFundExpense,
                net: _rangeFundNet,
                openingBalance: _fundSummary?.openingBalance ?? 0,
                closingBalance: _fundSummary?.closingBalance ?? _rangeFundNet,
                onEdit: _editFund,
                onDelete: _deleteFund,
                money: _money,
                dateText: _dateText,
              ),
            ],
            const SizedBox(height: 16),
            if (_loading)
              const Padding(
                padding: EdgeInsets.only(top: 80),
                child: Center(child: CircularProgressIndicator()),
              )
            else if (_error != null)
              _MessageCard(
                icon: Icons.error_outline_rounded,
                title: 'No se pudo cargar el historial',
                message: _error!,
                onRetry: _load,
              )
            else if (_items.isEmpty)
              const _MessageCard(
                icon: Icons.inventory_2_outlined,
                title: 'Sin cajas en este rango',
                message:
                    'No se encontraron cajas registradas para estas fechas.',
              )
            else
              ..._items.map(
                (cash) => _CashHistoryCard(
                  cash: cash,
                  money: _money,
                  dateText: _dateText,
                  statusLabel: _statusLabel,
                  statusColor: _statusColor,
                  onTap: () => _showCashDetail(cash),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _FundMovementHistoryCard extends StatefulWidget {
  final List<CashFundMovementModel> movements;
  final double totalIn;
  final double totalOut;
  final double net;
  final double openingBalance;
  final double closingBalance;
  final String Function(double) money;
  final String Function(String?) dateText;
  final ValueChanged<CashFundMovementModel> onEdit;
  final ValueChanged<CashFundMovementModel> onDelete;
  const _FundMovementHistoryCard({
    required this.movements,
    required this.totalIn,
    required this.totalOut,
    required this.net,
    required this.openingBalance,
    required this.closingBalance,
    required this.money,
    required this.dateText,
    required this.onEdit,
    required this.onDelete,
  });
  @override
  State<_FundMovementHistoryCard> createState() =>
      _FundMovementHistoryCardState();
}

class _FundMovementHistoryCardState extends State<_FundMovementHistoryCard> {
  bool _expanded = false;
  @override
  Widget build(BuildContext context) {
    if (_expanded) {
      return Column(children: [
        _FundMovementHistoryContent(
          movements: widget.movements,
          totalIn: widget.totalIn,
          totalOut: widget.totalOut,
          net: widget.net,
          openingBalance: widget.openingBalance,
          closingBalance: widget.closingBalance,
          money: widget.money,
          dateText: widget.dateText,
          onEdit: widget.onEdit,
          onDelete: widget.onDelete,
        ),
        TextButton.icon(
          onPressed: () => setState(() => _expanded = false),
          icon: const Icon(Icons.keyboard_arrow_up_rounded),
          label: const Text('Ocultar fondo acumulado'),
        ),
      ]);
    }
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () => setState(() => _expanded = true),
        borderRadius: BorderRadius.circular(22),
        child: Ink(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            gradient: const LinearGradient(
                colors: [Color(0xFF17131E), Color(0xFF261D32)]),
            borderRadius: BorderRadius.circular(22),
            border: Border.all(color: const Color(0x55D4A72C)),
          ),
          child: Row(children: [
            const Icon(Icons.savings_outlined, color: Color(0xFFF5C451)),
            const SizedBox(width: 12),
            Expanded(
                child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Fondo acumulado',
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w900)),
                const SizedBox(height: 3),
                Text(
                  'Saldo ${widget.money(widget.closingBalance)} · ${widget.movements.length} movimientos',
                  style: const TextStyle(
                      color: Color(0xFFCBD5E1),
                      fontSize: 11.5,
                      fontWeight: FontWeight.w700),
                ),
              ],
            )),
            const Icon(Icons.keyboard_arrow_down_rounded,
                color: Color(0xFFF5C451)),
          ]),
        ),
      ),
    );
  }
}

class _FundMovementHistoryContent extends StatelessWidget {
  final List<CashFundMovementModel> movements;
  final double totalIn;
  final double totalOut;
  final double net;
  final double openingBalance;
  final double closingBalance;
  final String Function(double) money;
  final String Function(String?) dateText;
  final ValueChanged<CashFundMovementModel> onEdit;
  final ValueChanged<CashFundMovementModel> onDelete;

  const _FundMovementHistoryContent({
    required this.movements,
    required this.totalIn,
    required this.totalOut,
    required this.net,
    required this.openingBalance,
    required this.closingBalance,
    required this.money,
    required this.dateText,
    required this.onEdit,
    required this.onDelete,
  });

  String _typeLabel(String type) => switch (type.toUpperCase()) {
        'CLOSING_DEPOSIT' => 'Ingreso por cierre',
        'OPENING_WITHDRAWAL' => 'Retiro para apertura',
        'MANUAL_DEPOSIT' => 'Ingreso manual',
        'MANUAL_WITHDRAWAL' => 'Retiro manual',
        'EXPENSE' => 'Gasto desde fondo',
        'ADJUSTMENT_IN' => 'Ajuste positivo',
        'ADJUSTMENT_OUT' => 'Ajuste negativo',
        _ => 'Movimiento de fondo',
      };

  String _methodLabel(String method) => switch (method.toUpperCase()) {
        'CASH' || 'EFECTIVO' => 'Efectivo',
        'CARD' || 'TARJETA' => 'Tarjeta',
        'TRANSFER' || 'TRANSFERENCIA' => 'Transferencia',
        'YAPE' => 'Yape',
        'PLIN' => 'Plin',
        _ => method,
      };

  @override
  Widget build(BuildContext context) {
    const gold = Color(0xFFD4A72C);
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF17131E), Color(0xFF261D32)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0x55D4A72C)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x2417131E),
            blurRadius: 20,
            offset: Offset(0, 9),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: gold.withValues(alpha: .16),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: gold.withValues(alpha: .35)),
                ),
                child: const Icon(
                  Icons.savings_outlined,
                  color: Color(0xFFF5C451),
                ),
              ),
              const SizedBox(width: 12),
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'MOVIMIENTOS DEL FONDO',
                      style: TextStyle(
                        fontSize: 9,
                        letterSpacing: 1.2,
                        fontWeight: FontWeight.w900,
                        color: Color(0xFFF5C451),
                      ),
                    ),
                    SizedBox(height: 2),
                    Text(
                      'Fondo acumulado',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w900,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: .08),
                  borderRadius: BorderRadius.circular(99),
                ),
                child: Text(
                  '${movements.length} mov.',
                  style: const TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w900,
                    color: Colors.white,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: _FundMetric(
                  label: 'INGRESOS',
                  value: money(totalIn),
                  color: const Color(0xFF6EE7B7),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _FundMetric(
                  label: 'RETIROS',
                  value: money(totalOut),
                  color: const Color(0xFFFDA29B),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _FundMetric(
                  label: 'VARIACIÓN',
                  value: money(net),
                  color: net >= 0
                      ? const Color(0xFFF5C451)
                      : const Color(0xFFFDA29B),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          const Text(
            'SALDO DEL RANGO · INICIAL + MOVIMIENTOS = FINAL',
            style: TextStyle(
              fontSize: 9,
              letterSpacing: .8,
              fontWeight: FontWeight.w900,
              color: Color(0xFFCBD5E1),
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: _FundMetric(
                  label: 'SALDO INICIAL',
                  value: money(openingBalance),
                  color: const Color(0xFFCBD5E1),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _FundMetric(
                  label: 'MOV. DEL RANGO',
                  value: '${net >= 0 ? '+' : ''}${money(net)}',
                  color: net >= 0
                      ? const Color(0xFFF5C451)
                      : const Color(0xFFFDA29B),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _FundMetric(
                  label: 'SALDO FINAL',
                  value: money(closingBalance),
                  color: const Color(0xFF6EE7B7),
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          if (movements.isEmpty)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(15),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: .06),
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Text(
                'No hubo ingresos ni retiros del fondo en este rango.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Color(0xFFCBD5E1),
                  fontWeight: FontWeight.w700,
                ),
              ),
            )
          else
            ...movements.map((movement) {
              final income = movement.isIncome;
              final editable = const {
                    'MANUAL_DEPOSIT',
                    'MANUAL_WITHDRAWAL',
                    'ADJUSTMENT_IN',
                    'ADJUSTMENT_OUT'
                  }.contains(movement.type.toUpperCase()) &&
                  movement.cashRegisterId == null;
              return Container(
                margin: const EdgeInsets.only(bottom: 8),
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: .07),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(
                    color: Colors.white.withValues(alpha: .08),
                  ),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: 34,
                      height: 34,
                      decoration: BoxDecoration(
                        color: (income
                                ? const Color(0xFF10B981)
                                : const Color(0xFFF04438))
                            .withValues(alpha: .16),
                        borderRadius: BorderRadius.circular(11),
                      ),
                      child: Icon(
                        income
                            ? Icons.south_west_rounded
                            : Icons.north_east_rounded,
                        size: 17,
                        color: income
                            ? const Color(0xFF6EE7B7)
                            : const Color(0xFFFDA29B),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            movement.concept,
                            style: const TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w900,
                              color: Colors.white,
                            ),
                          ),
                          const SizedBox(height: 3),
                          Text(
                            '${_typeLabel(movement.type)} · ${_methodLabel(movement.paymentMethod)}',
                            style: const TextStyle(
                              fontSize: 10.5,
                              fontWeight: FontWeight.w700,
                              color: Color(0xFFCBD5E1),
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            '${dateText(movement.movementDate)} · ${movement.actorUserName?.trim().isNotEmpty == true ? movement.actorUserName : 'Sistema'}${movement.cashRegisterId == null ? '' : ' · Caja #${movement.cashRegisterId}'}',
                            style: const TextStyle(
                              fontSize: 10,
                              fontWeight: FontWeight.w600,
                              color: Color(0xFF94A3B8),
                            ),
                          ),
                          if (movement.note?.trim().isNotEmpty == true) ...[
                            const SizedBox(height: 4),
                            Text(
                              movement.note!.trim(),
                              style: const TextStyle(
                                fontSize: 10.5,
                                color: Color(0xFFE2E8F0),
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text('${income ? '+' : '-'}${money(movement.amount)}',
                            style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w900,
                                color: income
                                    ? const Color(0xFF6EE7B7)
                                    : const Color(0xFFFDA29B))),
                        if (editable)
                          PopupMenuButton<String>(
                            iconColor: const Color(0xFFF5C451),
                            onSelected: (value) => value == 'edit'
                                ? onEdit(movement)
                                : onDelete(movement),
                            itemBuilder: (_) => const [
                              PopupMenuItem(
                                  value: 'edit', child: Text('Editar')),
                              PopupMenuItem(
                                  value: 'delete', child: Text('Eliminar'))
                            ],
                          ),
                      ],
                    ),
                  ],
                ),
              );
            }),
        ],
      ),
    );
  }
}

class _FundMetric extends StatelessWidget {
  final String label;
  final String value;
  final Color color;

  const _FundMetric({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: .06),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(
              fontSize: 8,
              letterSpacing: .7,
              fontWeight: FontWeight.w900,
              color: Color(0xFF94A3B8),
            ),
          ),
          const SizedBox(height: 4),
          FittedBox(
            fit: BoxFit.scaleDown,
            alignment: Alignment.centerLeft,
            child: Text(
              value,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w900,
                color: color,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _CashAuditActivitySheet extends StatefulWidget {
  final int branchId;
  final String branchName;
  final DateTime from;
  final DateTime to;
  final String Function(String?) dateText;

  const _CashAuditActivitySheet({
    required this.branchId,
    required this.branchName,
    required this.from,
    required this.to,
    required this.dateText,
  });

  @override
  State<_CashAuditActivitySheet> createState() =>
      _CashAuditActivitySheetState();
}

class _CashAuditActivitySheetState extends State<_CashAuditActivitySheet> {
  final OwnerCashService _service = OwnerCashService();

  bool _loading = true;
  String? _error;
  int? _actorUserId;
  List<CashAuditLogModel> _logs = [];
  List<_AuditActorOption> _actors = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load({int? actorUserId}) async {
    if (!mounted) return;
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final data = await _service.getCashAudit(
        branchId: widget.branchId,
        actorUserId: actorUserId,
        from: widget.from,
        to: widget.to,
      );
      if (!mounted) return;
      setState(() {
        _logs = data;
        _actors = _mergeActors(_actors, data);
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = SuperGodsUi.humanError(e);
        _loading = false;
      });
    }
  }

  List<_AuditActorOption> _mergeActors(
    List<_AuditActorOption> current,
    List<CashAuditLogModel> logs,
  ) {
    final map = <int, _AuditActorOption>{
      for (final actor in current) actor.id: actor,
    };
    for (final log in logs) {
      final id = log.actorUserId;
      if (id != null) {
        map[id] = _AuditActorOption(
          id: id,
          name: log.actorUserName?.trim().isNotEmpty == true
              ? log.actorUserName!.trim()
              : 'Usuario #$id',
        );
      }
    }
    final values = map.values.toList();
    values.sort((a, b) => a.name.compareTo(b.name));
    return values;
  }

  void _changeActor(int? actorUserId) {
    setState(() => _actorUserId = actorUserId);
    _load(actorUserId: actorUserId);
  }

  String _actionLabel(String action) {
    switch (action.toUpperCase()) {
      case 'UPDATE':
        return 'Edito';
      case 'DELETE':
        return 'Elimino';
      default:
        return action.isEmpty ? 'Cambio' : action;
    }
  }

  String _entityLabel(String entityType) {
    switch (entityType.toUpperCase()) {
      case 'SALE':
        return 'venta';
      case 'CASH_MOVEMENT':
        return 'movimiento';
      default:
        return 'registro';
    }
  }

  @override
  Widget build(BuildContext context) {
    final bottom = MediaQuery.of(context).viewInsets.bottom;

    return DraggableScrollableSheet(
      initialChildSize: 0.88,
      minChildSize: 0.55,
      maxChildSize: 0.95,
      builder: (context, controller) => Container(
        padding: EdgeInsets.fromLTRB(16, 10, 16, 16 + bottom),
        decoration: const BoxDecoration(
          color: Color(0xFFF6F3ED),
          borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        ),
        child: ListView(
          controller: controller,
          children: [
            Center(
              child: Container(
                width: 46,
                height: 5,
                decoration: BoxDecoration(
                  color: const Color(0xFFD0D5DD),
                  borderRadius: BorderRadius.circular(99),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Actividad de caja',
              style: const TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.w900,
                color: Color(0xFF17131E),
              ),
            ),
            const SizedBox(height: 4),
            Text(
              widget.branchName,
              style: const TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w800,
                color: Color(0xFF667085),
              ),
            ),
            const SizedBox(height: 14),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFFEBDFAF)),
              ),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<int?>(
                  value: _actorUserId,
                  isExpanded: true,
                  items: [
                    const DropdownMenuItem<int?>(
                      value: null,
                      child: Text('Todos los usuarios'),
                    ),
                    ..._actors.map(
                      (actor) => DropdownMenuItem<int?>(
                        value: actor.id,
                        child: Text(actor.name),
                      ),
                    ),
                  ],
                  onChanged: _changeActor,
                ),
              ),
            ),
            const SizedBox(height: 14),
            if (_loading)
              const Padding(
                padding: EdgeInsets.only(top: 60),
                child: Center(child: CircularProgressIndicator()),
              )
            else if (_error != null)
              _MessageCard(
                icon: Icons.error_outline_rounded,
                title: 'No se pudo cargar la actividad',
                message: _error!,
                onRetry: () => _load(actorUserId: _actorUserId),
              )
            else if (_logs.isEmpty)
              const _MessageCard(
                icon: Icons.manage_search_rounded,
                title: 'Sin actividad',
                message: 'No hay cambios auditados para este rango.',
              )
            else
              ..._logs.map(
                (log) => _AuditActivityCard(
                  log: log,
                  dateText: widget.dateText,
                  title:
                      '${_actionLabel(log.action)} ${_entityLabel(log.entityType)} #${log.entityId ?? '-'}',
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _AuditActorOption {
  final int id;
  final String name;

  const _AuditActorOption({required this.id, required this.name});
}

class _AuditActivityCard extends StatelessWidget {
  final CashAuditLogModel log;
  final String Function(String?) dateText;
  final String title;

  const _AuditActivityCard({
    required this.log,
    required this.dateText,
    required this.title,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFF0E7D3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w900,
              color: Color(0xFF17131E),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Caja #${log.cashRegisterId ?? '-'} - ${dateText(log.createdAt)} - ${log.actorUserName ?? 'Sistema'}',
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              color: Color(0xFF667085),
            ),
          ),
          if (log.reason?.trim().isNotEmpty == true) ...[
            const SizedBox(height: 8),
            Text(
              'Motivo: ${log.reason!.trim()}',
              style: const TextStyle(
                fontSize: 12.5,
                fontWeight: FontWeight.w800,
                color: Color(0xFF8B4E00),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _HeaderCard extends StatelessWidget {
  final String branchName;
  final String from;
  final String to;
  final VoidCallback onFrom;
  final VoidCallback onTo;
  final VoidCallback onActivity;

  const _HeaderCard({
    required this.branchName,
    required this.from,
    required this.to,
    required this.onFrom,
    required this.onTo,
    required this.onActivity,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: const Color(0xFFEBDFAF)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 12,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Sede',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w800,
              color: Color(0xFF9C8E6A),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            branchName,
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w900,
              color: Color(0xFF17131E),
            ),
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: _DateButton(
                  label: 'Desde',
                  value: from,
                  onTap: onFrom,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _DateButton(
                  label: 'Hasta',
                  value: to,
                  onTap: onTo,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: onActivity,
              icon: const Icon(Icons.manage_search_rounded),
              label: const Text('Actividad reciente de caja'),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF17131E),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                textStyle: const TextStyle(fontWeight: FontWeight.w900),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _DateButton extends StatelessWidget {
  final String label;
  final String value;
  final VoidCallback onTap;

  const _DateButton({
    required this.label,
    required this.value,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(16),
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: const Color(0xFFF8F4EA),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: const Color(0xFFEBDFAF)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: const TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w800,
                color: Color(0xFF8B7A4E),
              ),
            ),
            const SizedBox(height: 4),
            Row(
              children: [
                Expanded(
                  child: Text(
                    value,
                    style: const TextStyle(
                      fontSize: 13.5,
                      fontWeight: FontWeight.w900,
                      color: Color(0xFF17131E),
                    ),
                  ),
                ),
                const Icon(
                  Icons.calendar_month_rounded,
                  size: 18,
                  color: Color(0xFF8B7A4E),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _CashHistoryCard extends StatelessWidget {
  final CashRegisterModel cash;
  final String Function(double) money;
  final String Function(String?) dateText;
  final String Function(String) statusLabel;
  final Color Function(String) statusColor;
  final VoidCallback onTap;

  const _CashHistoryCard({
    required this.cash,
    required this.money,
    required this.dateText,
    required this.statusLabel,
    required this.statusColor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final color = statusColor(cash.status);
    final payments = cash.paymentMethodsSummary;

    return InkWell(
      borderRadius: BorderRadius.circular(20),
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: const Color(0xFFF0E7D3)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.045),
              blurRadius: 12,
              offset: const Offset(0, 5),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: color.withOpacity(0.10),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Icon(
                Icons.point_of_sale_rounded,
                color: color,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    dateText(cash.openedAt),
                    style: const TextStyle(
                      fontSize: 14.5,
                      fontWeight: FontWeight.w900,
                      color: Color(0xFF17131E),
                    ),
                  ),
                  const SizedBox(height: 5),
                  Text(
                    'Ventas totales: ${money(cash.salesTotal)}',
                    style: const TextStyle(
                      fontSize: 12.5,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFF6B7280),
                    ),
                  ),
                  if (payments.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    _PaymentMethodsWrap(
                      payments: payments,
                      money: money,
                      compact: true,
                    ),
                  ],
                  const SizedBox(height: 5),
                  Text(
                    'Esperado: ${money(cash.closingAmountExpected)}',
                    style: const TextStyle(
                      fontSize: 12.5,
                      fontWeight: FontWeight.w800,
                      color: Color(0xFF7A5A22),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.10),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Text(
                    statusLabel(cash.status),
                    style: TextStyle(
                      color: color,
                      fontSize: 11,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                const Icon(
                  Icons.chevron_right_rounded,
                  color: Color(0xFF98A2B3),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _SaleItemEditDraft {
  _SaleItemEditDraft({required this.item})
      : barberUserId = item.barberUserId,
        priceController = TextEditingController(
          text: item.precioUnitario.toStringAsFixed(2),
        );

  final CashSaleItemModel item;
  int? barberUserId;
  final TextEditingController priceController;

  double get price =>
      double.tryParse(priceController.text.trim().replaceAll(',', '.')) ??
      item.precioUnitario;

  void dispose() => priceController.dispose();
}

class _CashHistoryDetailSheet extends StatefulWidget {
  final CashRegisterModel cash;
  final int branchId;
  final DateTime from;
  final DateTime to;
  final String Function(double) money;
  final String Function(String?) dateText;
  final String Function(String) statusLabel;
  final Color Function(String) statusColor;
  final bool canManageFund;

  const _CashHistoryDetailSheet({
    required this.cash,
    required this.branchId,
    required this.from,
    required this.to,
    required this.money,
    required this.dateText,
    required this.statusLabel,
    required this.statusColor,
    required this.canManageFund,
  });

  @override
  State<_CashHistoryDetailSheet> createState() =>
      _CashHistoryDetailSheetState();
}

class _CashHistoryDetailSheetState extends State<_CashHistoryDetailSheet> {
  final OwnerCashService _service = OwnerCashService();
  final OwnerSaleCatalogService _catalogService = OwnerSaleCatalogService();

  bool _loadingSales = true;
  String? _salesError;
  List<CashSaleModel> _sales = [];
  late List<CashMovementModel> _movements;
  List<SaleBarberModel> _barbers = [];
  String _selectedSalesBarber = 'ALL';
  bool _loadingAudit = true;
  String? _auditError;
  List<CashAuditLogModel> _auditLogs = [];

  CashRegisterModel get cash => widget.cash;
  String money(double value) => widget.money(value);
  String dateText(String? value) => widget.dateText(value);

  bool get _isCashClosed {
    final status = cash.status.trim().toUpperCase();
    return status == 'CLOSED' || status == 'AUTO_CLOSED';
  }

  double get _calculatedDifference =>
      cash.closingAmountCounted - cash.closingAmountExpected;

  DateTime get _cashDate {
    final openedAt = DateTime.tryParse(cash.openedAt?.trim() ?? '');
    final fallback =
        DateTime(widget.from.year, widget.from.month, widget.from.day);
    return openedAt ?? fallback;
  }

  DateTime get _requestedStart =>
      DateTime(widget.from.year, widget.from.month, widget.from.day);
  DateTime get _requestedEnd =>
      DateTime(widget.to.year, widget.to.month, widget.to.day)
          .add(const Duration(days: 1));

  bool get _cashDateIsInsideRequestedRange {
    final day = DateTime(_cashDate.year, _cashDate.month, _cashDate.day);
    return !day.isBefore(_requestedStart) && day.isBefore(_requestedEnd);
  }

  DateTime get _filterStart => _cashDateIsInsideRequestedRange
      ? DateTime(_cashDate.year, _cashDate.month, _cashDate.day)
      : _requestedStart;

  DateTime get _filterEnd => _cashDateIsInsideRequestedRange
      ? _filterStart.add(const Duration(days: 1))
      : _requestedEnd;

  List<CashMovementModel> get _visibleMovements => _movements
      .where((movement) =>
          _isDateInFilter(movement.movementDate) &&
          (widget.canManageFund ||
              movement.fundingSource != 'ACCUMULATED_FUND'))
      .toList();

  Map<String, double> _movementTotalsByMethod(String type) {
    final result = <String, double>{};
    for (final movement in _visibleMovements.where(
      (movement) => movement.type.toUpperCase() == type,
    )) {
      final method = normalizePaymentCode(movement.paymentMethod ?? 'CASH');
      result[method] = (result[method] ?? 0) + movement.amount;
    }
    return result;
  }

  List<CashMovementModel> _movementsOfType(String type) => _visibleMovements
      .where((movement) => movement.type.toUpperCase() == type)
      .toList();

  List<CashMovementModel> get _expenseDetails => _movementsOfType('EXPENSE');

  List<CashMovementModel> get _advanceDetails =>
      _movementsOfType('ADVANCE_BARBER');

  List<CashMovementModel> get _barberPaymentDetails =>
      _movementsOfType('PAYMENT_BARBER');

  double _totalMovements(List<CashMovementModel> movements) =>
      movements.fold<double>(0, (sum, movement) => sum + movement.amount);

  double get _expenseTotalForCash => _totalMovements(_expenseDetails);
  double get _advanceTotalForCash => _totalMovements(_advanceDetails);
  double get _barberPaymentTotalForCash =>
      _totalMovements(_barberPaymentDetails);

  Map<String, double> get _expensesByMethod =>
      _movementTotalsByMethod('EXPENSE');

  Map<String, double> get _advancesByMethod =>
      _movementTotalsByMethod('ADVANCE_BARBER');

  Map<String, double> get _barberPaymentsByMethod =>
      _movementTotalsByMethod('PAYMENT_BARBER');

  List<String> get _salesBarberOptions {
    final names = <String>{};
    for (final sale in _sales) {
      final name = _readSaleBarber(sale);
      if (name != 'Profesional no registrado') names.add(name);
    }
    return names.toList()..sort();
  }

  List<CashSaleModel> get _filteredSales {
    if (_selectedSalesBarber == 'ALL') return _sales;
    return _sales
        .where(
          (sale) => _readSaleBarber(sale) == _selectedSalesBarber,
        )
        .toList();
  }

  bool _isDateInFilter(String? value) {
    if (value == null || value.trim().isEmpty) return false;
    final parsed = DateTime.tryParse(value.trim());
    if (parsed == null) return false;
    return !parsed.isBefore(_filterStart) && parsed.isBefore(_filterEnd);
  }

  @override
  void initState() {
    super.initState();
    _movements = List<CashMovementModel>.from(cash.movements);
    _loadBarbers();
    _loadSales();
  }

  Future<void> _loadBarbers() async {
    try {
      final data = await _catalogService.getBarbers(branchId: widget.branchId);
      if (!mounted) return;
      setState(() => _barbers = data);
    } catch (_) {
      if (!mounted) return;
      setState(() => _barbers = const []);
    }
  }

  Future<void> _loadSales() async {
    if (!mounted) return;

    setState(() {
      _loadingSales = true;
      _salesError = null;
    });

    try {
      final data = await _service.getSalesByCashRegister(
        branchId: widget.branchId,
        cashRegisterId: cash.id,
      );

      if (!mounted) return;
      setState(() {
        _sales =
            data.where((sale) => _isDateInFilter(sale.fechaCreacion)).toList();
        _loadingSales = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _salesError = SuperGodsUi.humanError(e);
        _loadingSales = false;
      });
    }
  }

  Future<void> _loadAudit() async {
    if (!mounted) return;

    setState(() {
      _loadingAudit = true;
      _auditError = null;
    });

    try {
      final now = DateTime.now();
      final data = await _service.getCashAudit(
        branchId: widget.branchId,
        cashRegisterId: cash.id,
        from: _filterStart,
        to: DateTime(now.year, now.month, now.day),
      );

      if (!mounted) return;
      setState(() {
        _auditLogs = data;
        _loadingAudit = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _auditError = SuperGodsUi.humanError(e);
        _loadingAudit = false;
      });
    }
  }

  Future<String?> _requestAuditReason(String title) async {
    await WidgetsBinding.instance.endOfFrame;
    if (!mounted) return null;
    return SuperGodsUi.requestReason(
      context,
      title: title,
      hint: 'Ej. Corrección de monto, fecha o profesional.',
    );
  }

  double _parseMoney(String value, double fallback) {
    return double.tryParse(value.trim().replaceAll(',', '.')) ?? fallback;
  }

  Future<void> _handleEditSale(CashSaleModel sale) async {
    final subtotalCtrl =
        TextEditingController(text: sale.subtotal.toStringAsFixed(2));
    final discountCtrl =
        TextEditingController(text: sale.discount.toStringAsFixed(2));
    final totalCtrl =
        TextEditingController(text: sale.total.toStringAsFixed(2));
    final cashReceivedCtrl =
        TextEditingController(text: sale.cashReceived.toStringAsFixed(2));
    final drafts =
        sale.items.map((item) => _SaleItemEditDraft(item: item)).toList();

    final ok = await showDialog<bool>(
      context: context,
      barrierColor: Colors.black.withValues(alpha: .55),
      builder: (_) => StatefulBuilder(
        builder: (context, setDialogState) {
          const ink = Color(0xFF17131E);
          const muted = Color(0xFF667085);
          const gold = Color(0xFFD6B36A);
          const border = Color(0xFFE9DFCB);

          InputDecoration fieldDecoration(String label, {IconData? icon}) {
            return SuperGodsUi.inputDecoration(label: label, icon: icon);
          }

          Widget moneyField({
            required TextEditingController controller,
            required String label,
            IconData? icon,
          }) {
            return TextField(
              controller: controller,
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              style: TextStyle(
                color: ink,
                fontSize: 14,
                fontWeight: FontWeight.w900,
              ),
              decoration: fieldDecoration(label, icon: icon),
            );
          }

          return Theme(
            data: Theme.of(context).copyWith(
              brightness: Brightness.light,
              canvasColor: Colors.white,
              dialogBackgroundColor: Colors.transparent,
              colorScheme: Theme.of(context).colorScheme.copyWith(
                    brightness: Brightness.light,
                    primary: ink,
                    secondary: gold,
                    surface: Colors.white,
                  ),
            ),
            child: Dialog(
              insetPadding:
                  const EdgeInsets.symmetric(horizontal: 18, vertical: 24),
              backgroundColor: Colors.transparent,
              child: Container(
                constraints: const BoxConstraints(maxWidth: 430),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(28),
                  border: Border.all(color: const Color(0xFFF0E7D3)),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.24),
                      blurRadius: 32,
                      offset: const Offset(0, 18),
                    ),
                  ],
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(28),
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.fromLTRB(18, 18, 18, 16),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Container(
                              width: 42,
                              height: 42,
                              decoration: BoxDecoration(
                                color: const Color(0xFF17131E),
                                borderRadius: BorderRadius.circular(16),
                                boxShadow: [
                                  BoxShadow(
                                    color: gold.withOpacity(0.22),
                                    blurRadius: 16,
                                    offset: const Offset(0, 8),
                                  ),
                                ],
                              ),
                              child: Icon(
                                Icons.workspace_premium_outlined,
                                color: gold,
                                size: 22,
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    'Editar venta VIP',
                                    style: TextStyle(
                                      color: ink,
                                      fontSize: 20,
                                      fontWeight: FontWeight.w900,
                                    ),
                                  ),
                                  SizedBox(height: 2),
                                  Text(
                                    'Historial de caja',
                                    style: TextStyle(
                                      color: muted,
                                      fontSize: 12.5,
                                      fontWeight: FontWeight.w800,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 18),
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: const Color(0xFFFCFBF8),
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(color: border),
                          ),
                          child: Column(
                            children: [
                              Row(
                                children: [
                                  Expanded(
                                    child: moneyField(
                                      controller: subtotalCtrl,
                                      label: 'Subtotal',
                                      icon: Icons.receipt_long_outlined,
                                    ),
                                  ),
                                  const SizedBox(width: 10),
                                  Expanded(
                                    child: moneyField(
                                      controller: discountCtrl,
                                      label: 'Descuento',
                                      icon: Icons.local_offer_outlined,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 10),
                              Row(
                                children: [
                                  Expanded(
                                    child: moneyField(
                                      controller: totalCtrl,
                                      label: 'Total',
                                      icon: Icons.payments_outlined,
                                    ),
                                  ),
                                  const SizedBox(width: 10),
                                  Expanded(
                                    child: moneyField(
                                      controller: cashReceivedCtrl,
                                      label: 'Recibido',
                                      icon: Icons.point_of_sale_outlined,
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 14),
                        ...drafts.map((draft) {
                          final item = draft.item;
                          final name = (item.serviceName?.trim().isNotEmpty ==
                                  true)
                              ? item.serviceName!.trim()
                              : (item.productName?.trim().isNotEmpty == true)
                                  ? item.productName!.trim()
                                  : 'Item';
                          return Container(
                            margin: const EdgeInsets.only(bottom: 12),
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(20),
                              border: Border.all(color: border),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withOpacity(0.035),
                                  blurRadius: 14,
                                  offset: const Offset(0, 8),
                                ),
                              ],
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Container(
                                      width: 34,
                                      height: 34,
                                      decoration: BoxDecoration(
                                        color: const Color(0xFFFFF7E0),
                                        borderRadius: BorderRadius.circular(13),
                                      ),
                                      child: const Icon(
                                        Icons.content_cut,
                                        color: Color(0xFFC48919),
                                        size: 18,
                                      ),
                                    ),
                                    const SizedBox(width: 10),
                                    Expanded(
                                      child: Text(
                                        '$name x${item.cantidad}',
                                        maxLines: 2,
                                        overflow: TextOverflow.ellipsis,
                                        style: TextStyle(
                                          color: ink,
                                          fontSize: 14,
                                          fontWeight: FontWeight.w900,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 12),
                                DropdownButtonFormField<int?>(
                                  value: draft.barberUserId,
                                  isExpanded: true,
                                  menuMaxHeight: 260,
                                  dropdownColor: Colors.white,
                                  borderRadius: BorderRadius.circular(18),
                                  icon: const Icon(Icons.keyboard_arrow_down),
                                  style: TextStyle(
                                    color: ink,
                                    fontSize: 13.5,
                                    fontWeight: FontWeight.w900,
                                  ),
                                  decoration: fieldDecoration(
                                    'Profesional',
                                    icon: Icons.person_outline,
                                  ),
                                  items: [
                                    const DropdownMenuItem<int?>(
                                      value: null,
                                      child: Text('Sin profesional'),
                                    ),
                                    ..._barbers.map(
                                      (barber) => DropdownMenuItem<int?>(
                                        value: barber.id,
                                        child: Text(
                                          barber.nombre,
                                          overflow: TextOverflow.ellipsis,
                                        ),
                                      ),
                                    ),
                                  ],
                                  onChanged: (value) => setDialogState(
                                      () => draft.barberUserId = value),
                                ),
                                const SizedBox(height: 10),
                                moneyField(
                                  controller: draft.priceController,
                                  label: 'Precio unitario',
                                  icon: Icons.sell_outlined,
                                ),
                              ],
                            ),
                          );
                        }),
                        const SizedBox(height: 2),
                        Row(
                          children: [
                            Expanded(
                              child: OutlinedButton(
                                onPressed: () => Navigator.pop(context, false),
                                style: OutlinedButton.styleFrom(
                                  foregroundColor: ink,
                                  side: const BorderSide(color: border),
                                  padding:
                                      const EdgeInsets.symmetric(vertical: 14),
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(18),
                                  ),
                                ),
                                child: const Text(
                                  'Cancelar',
                                  style: TextStyle(fontWeight: FontWeight.w900),
                                ),
                              ),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: ElevatedButton(
                                onPressed: () => Navigator.pop(context, true),
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: ink,
                                  foregroundColor: Colors.white,
                                  elevation: 0,
                                  padding:
                                      const EdgeInsets.symmetric(vertical: 14),
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(18),
                                  ),
                                ),
                                child: const Text(
                                  'Guardar',
                                  style: TextStyle(fontWeight: FontWeight.w900),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );

    if (ok != true) {
      subtotalCtrl.dispose();
      discountCtrl.dispose();
      totalCtrl.dispose();
      cashReceivedCtrl.dispose();
      for (final draft in drafts) {
        draft.dispose();
      }
      return;
    }

    final auditReason = await _requestAuditReason('Motivo de edicion');
    if (auditReason == null) {
      subtotalCtrl.dispose();
      discountCtrl.dispose();
      totalCtrl.dispose();
      cashReceivedCtrl.dispose();
      for (final draft in drafts) {
        draft.dispose();
      }
      return;
    }

    try {
      final cashPayment = sale.payments
          .where((p) => normalizePaymentCode(p.method) == 'CASH')
          .fold<double>(0, (sum, p) => sum + p.amount);
      final cashReceived =
          _parseMoney(cashReceivedCtrl.text, sale.cashReceived);
      final change = cashPayment > 0
          ? (cashReceived - cashPayment).clamp(0, double.infinity).toDouble()
          : sale.changeAmount;

      await _service.updateSale(
        branchId: widget.branchId,
        saleId: sale.saleId,
        metodoPago: sale.metodoPago,
        subtotal: _parseMoney(subtotalCtrl.text, sale.subtotal),
        discount: _parseMoney(discountCtrl.text, sale.discount),
        total: _parseMoney(totalCtrl.text, sale.total),
        cashReceived: cashReceived,
        changeAmount: change,
        customerId: sale.customerId,
        auditReason: auditReason,
        payments: sale.payments
            .map((payment) => CashSalePaymentPayload(
                  method: payment.method,
                  amount: payment.amount,
                ))
            .toList(),
        items: drafts
            .map((draft) => CashSaleItemUpdatePayload(
                  saleItemId: draft.item.saleItemId,
                  barberUserId: draft.barberUserId,
                  precioUnitario: draft.price,
                ))
            .toList(),
      );

      await _loadSales();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Venta actualizada correctamente.')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(SuperGodsUi.humanError(e))),
      );
    } finally {
      subtotalCtrl.dispose();
      discountCtrl.dispose();
      totalCtrl.dispose();
      cashReceivedCtrl.dispose();
      for (final draft in drafts) {
        draft.dispose();
      }
    }
  }

  Future<void> _handleDeleteSale(CashSaleModel sale) async {
    final ok = await SuperGodsUi.confirm(
      context,
      title: 'Eliminar venta',
      message:
          'La venta será retirada y la acción quedará protegida por auditoría. No se puede deshacer.',
      confirmLabel: 'Eliminar',
      icon: Icons.delete_outline_rounded,
      destructive: true,
    );

    if (ok != true) return;

    final auditReason = await _requestAuditReason('Motivo de eliminacion');
    if (auditReason == null) return;

    try {
      await _service.deleteSale(
        branchId: widget.branchId,
        saleId: sale.saleId,
        auditReason: auditReason,
      );
      await _loadSales();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Venta eliminada correctamente.')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(SuperGodsUi.humanError(e))),
      );
    }
  }

  Future<void> _handleEditMovement(CashMovementModel movement) async {
    if (movement.fundingSource == 'ACCUMULATED_FUND' && !widget.canManageFund) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
              'No tienes permiso para modificar movimientos del fondo acumulado.'),
        ),
      );
      return;
    }
    final updated = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => OwnerCashMovementSheet(
        branchId: widget.branchId,
        branchName: cash.branchName ?? 'Sede',
        cashRegisterId: cash.id,
        initialMovement: movement,
        canManageFund: widget.canManageFund,
      ),
    );

    if (updated == true) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Movimiento actualizado correctamente.')),
      );
      Navigator.of(context).pop();
    }
  }

  Future<void> _handleDeleteMovement(CashMovementModel movement) async {
    if (movement.fundingSource == 'ACCUMULATED_FUND' && !widget.canManageFund) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
              'No tienes permiso para eliminar movimientos del fondo acumulado.'),
        ),
      );
      return;
    }
    final ok = await SuperGodsUi.confirm(
      context,
      title: 'Eliminar movimiento',
      message:
          'El movimiento será retirado y la acción quedará protegida por auditoría. No se puede deshacer.',
      confirmLabel: 'Eliminar',
      icon: Icons.delete_outline_rounded,
      destructive: true,
    );

    if (ok != true) return;

    final auditReason = await _requestAuditReason('Motivo de eliminacion');
    if (auditReason == null) return;

    try {
      await _service.deleteCashMovement(
        branchId: widget.branchId,
        movementId: movement.id,
        auditReason: auditReason,
      );
      if (!mounted) return;
      setState(() {
        _movements = _movements.where((m) => m.id != movement.id).toList();
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Movimiento eliminado correctamente.')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(SuperGodsUi.humanError(e))),
      );
    }
  }

  String _movementTypeLabel(String type) {
    switch (type.toUpperCase()) {
      case 'EXPENSE':
        return 'Gasto';
      case 'ADVANCE_BARBER':
        return 'Adelanto';
      case 'PAYMENT_BARBER':
        return 'Pago profesional';
      case 'INCOME':
        return 'Ingreso';
      case 'ADJUSTMENT':
        return 'Ajuste';
      default:
        return type;
    }
  }

  String _readSaleCustomer(CashSaleModel sale) {
    final name = sale.customerName?.trim();
    if (name != null && name.isNotEmpty) return name;
    return 'Cliente ocasional';
  }

  String _readSaleBarber(CashSaleModel sale) {
    final itemBarbers = <String>[];

    for (final item in sale.items) {
      final candidate = item.barberUserName?.trim();
      if (candidate != null &&
          candidate.isNotEmpty &&
          !itemBarbers.contains(candidate)) {
        itemBarbers.add(candidate);
      }
    }

    if (itemBarbers.isNotEmpty) return itemBarbers.join(', ');

    final direct = sale.barberName?.trim();
    if (direct != null && direct.isNotEmpty) return direct;

    return 'Profesional no registrado';
  }

  @override
  Widget build(BuildContext context) {
    final color = widget.statusColor(cash.status);

    return DraggableScrollableSheet(
      initialChildSize: 0.86,
      minChildSize: 0.45,
      maxChildSize: 0.96,
      builder: (_, controller) => Container(
        padding: const EdgeInsets.fromLTRB(16, 10, 16, 16),
        decoration: const BoxDecoration(
          color: Color(0xFFF6F3ED),
          borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        ),
        child: ListView(
          controller: controller,
          children: [
            Center(
              child: Container(
                width: 46,
                height: 5,
                decoration: BoxDecoration(
                  color: const Color(0xFFD0D5DD),
                  borderRadius: BorderRadius.circular(99),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                const Expanded(
                  child: Text(
                    'Detalle de caja',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.w900,
                      color: Color(0xFF17131E),
                    ),
                  ),
                ),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.10),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Text(
                    widget.statusLabel(cash.status),
                    style: TextStyle(
                      color: color,
                      fontSize: 12,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            _DetailBlock(
              title: 'Apertura y cierre',
              children: [
                _DetailRow('Abierta', dateText(cash.openedAt)),
                _DetailRow('Cerrada', dateText(cash.closedAt)),
                _DetailRow('Abrió', cash.openedByUserName ?? 'No registrado'),
                _DetailRow(
                    'Responsable', cash.assignedUserName ?? 'No asignado'),
              ],
            ),
            const SizedBox(height: 12),
            _DetailBlock(
              title: 'Resumen',
              children: [
                _DetailRow('Monto apertura', money(cash.openingAmount)),
                _DetailRow('Ventas totales', money(cash.salesTotal)),
                _DetailRow('Ventas efectivo', money(cash.cashSalesTotal)),
                _DetailRow('Ingresos caja', money(cash.movementsIncome)),
                _DetailRow('Gastos', money(_expenseTotalForCash)),
                if (_expensesByMethod.isNotEmpty)
                  _MovementMethodsBreakdown(
                    title: 'Detalle de gastos por método',
                    values: _expensesByMethod,
                    details: _expenseDetails,
                    money: money,
                    tone: const Color(0xFFB54708),
                  ),
                _DetailRow('Adelantos', money(_advanceTotalForCash)),
                if (_advancesByMethod.isNotEmpty)
                  _MovementMethodsBreakdown(
                    title: 'Detalle de adelantos por método',
                    values: _advancesByMethod,
                    details: _advanceDetails,
                    money: money,
                    tone: const Color(0xFF7A5AF8),
                  ),
                _DetailRow(
                  'Pagos profesional',
                  money(_barberPaymentTotalForCash),
                ),
                if (_barberPaymentsByMethod.isNotEmpty)
                  _MovementMethodsBreakdown(
                    title: 'Pagos a profesionales por método',
                    values: _barberPaymentsByMethod,
                    details: _barberPaymentDetails,
                    money: money,
                    tone: const Color(0xFF027A48),
                  ),
                _DetailRow(
                  'Efectivo esperado',
                  money(cash.closingAmountExpected),
                ),
                _DetailRow(
                  'Efectivo contado',
                  _isCashClosed
                      ? money(cash.closingAmountCounted)
                      : 'Pendiente de cierre',
                ),
                _DetailRow(
                  'Diferencia de arqueo',
                  _isCashClosed
                      ? money(_calculatedDifference)
                      : 'Pendiente de cierre',
                ),
              ],
            ),
            if (cash.paymentMethodsSummary.isNotEmpty) ...[
              const SizedBox(height: 12),
              _DetailBlock(
                title: 'Métodos de pago',
                children: [
                  _PaymentMethodsWrap(
                    payments: cash.paymentMethodsSummary,
                    money: money,
                    compact: false,
                  ),
                ],
              ),
            ],
            const SizedBox(height: 12),
            _DetailBlock(
              title: 'Ventas detalle',
              children: [
                if (!_loadingSales && _salesBarberOptions.isNotEmpty) ...[
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.fromLTRB(10, 10, 10, 8),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF7F4EE),
                      borderRadius: BorderRadius.circular(15),
                      border: Border.all(color: const Color(0xFFE8DECB)),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Row(
                          children: [
                            Icon(
                              Icons.content_cut_rounded,
                              size: 16,
                              color: Color(0xFF7A5A22),
                            ),
                            SizedBox(width: 7),
                            Text(
                              'Filtrar ventas por profesional',
                              style: TextStyle(
                                color: Color(0xFF475467),
                                fontSize: 11.5,
                                fontWeight: FontWeight.w900,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        SizedBox(
                          height: 38,
                          child: ListView(
                            scrollDirection: Axis.horizontal,
                            children: [
                              ChoiceChip(
                                label: const Text('Todos'),
                                selected: _selectedSalesBarber == 'ALL',
                                onSelected: (_) => setState(
                                  () => _selectedSalesBarber = 'ALL',
                                ),
                              ),
                              const SizedBox(width: 7),
                              ..._salesBarberOptions.map(
                                (name) => Padding(
                                  padding: const EdgeInsets.only(right: 7),
                                  child: ChoiceChip(
                                    label: Text(name),
                                    selected: _selectedSalesBarber == name,
                                    onSelected: (_) => setState(
                                      () => _selectedSalesBarber = name,
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 12),
                ],
                if (_loadingSales)
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 14),
                    child: Center(child: CircularProgressIndicator()),
                  )
                else if (_salesError != null)
                  _InlineError(message: _salesError!, onRetry: _loadSales)
                else if (_sales.isEmpty)
                  const _DetailRow('Sin ventas', 'No hubo ventas en esta caja.')
                else if (_filteredSales.isEmpty)
                  const _DetailRow(
                    'Sin resultados',
                    'No hay ventas de este profesional.',
                  )
                else
                  ..._filteredSales.map((sale) => _SaleDetailCard(
                        sale: sale,
                        money: money,
                        dateText: dateText,
                        customerName: _readSaleCustomer(sale),
                        barberName: _readSaleBarber(sale),
                        onEdit: () => _handleEditSale(sale),
                        onDelete: () => _handleDeleteSale(sale),
                      )),
              ],
            ),
            const SizedBox(height: 12),
            _DetailBlock(
              title: 'Gastos, adelantos y pagos',
              children: _visibleMovements.isEmpty
                  ? const [
                      _DetailRow('Sin movimientos',
                          'No hubo gastos, adelantos ni pagos en este rango.'),
                    ]
                  : _visibleMovements
                      .map(
                        (m) => _MovementDetailCard(
                          title: m.concept,
                          subtitle:
                              '${_movementTypeLabel(m.type)} · ${m.paymentMethod ?? 'CASH'}',
                          amount: money(m.amount),
                          date: dateText(m.movementDate),
                          barberName: m.barberUserName,
                          note: m.note,
                          isIncome: m.type.toUpperCase() == 'INCOME' ||
                              m.type.toUpperCase() == 'ADJUSTMENT',
                          onEdit: () => _handleEditMovement(m),
                          onDelete: () => _handleDeleteMovement(m),
                        ),
                      )
                      .toList(),
            ),
            const SizedBox(height: 12),
            _DetailBlock(
              title: 'Actividad de auditoria',
              children: [
                if (_loadingAudit)
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 14),
                    child: Center(child: CircularProgressIndicator()),
                  )
                else if (_auditError != null)
                  _InlineError(
                    message: _auditError!,
                    onRetry: _loadAudit,
                    retryLabel: 'Reintentar actividad',
                  )
                else if (_auditLogs.isEmpty)
                  const _DetailRow('Sin cambios',
                      'Aun no hay ediciones o eliminaciones auditadas.')
                else
                  ..._auditLogs.map(
                    (log) => _AuditLogCard(
                      log: log,
                      dateText: dateText,
                    ),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _MovementMethodsBreakdown extends StatelessWidget {
  final String title;
  final Map<String, double> values;
  final List<CashMovementModel> details;
  final Color tone;
  final String Function(double) money;

  const _MovementMethodsBreakdown({
    required this.title,
    required this.values,
    required this.details,
    required this.tone,
    required this.money,
  });

  String _label(String code) => switch (code.toUpperCase()) {
        'CASH' => 'Efectivo',
        'YAPE' => 'Yape',
        'PLIN' => 'Plin',
        'CARD' => 'Tarjeta',
        'TRANSFER' => 'Transferencia',
        _ => code,
      };

  @override
  Widget build(BuildContext context) {
    final entries = values.entries.where((entry) => entry.value > 0).toList();
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(top: 5, bottom: 8),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF8E8),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFEBDFAF)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: TextStyle(
              fontSize: 11.5,
              fontWeight: FontWeight.w900,
              color: tone,
            ),
          ),
          const SizedBox(height: 7),
          ...entries.map(
            (entry) => _DetailRow(
              _label(entry.key),
              '- ${money(entry.value)}',
            ),
          ),
          if (details.isNotEmpty) ...[
            const Divider(height: 15),
            ...details.map((movement) {
              final professional = movement.barberUserName?.trim();
              final note = movement.note?.trim();
              final method = _label(
                normalizePaymentCode(movement.paymentMethod ?? 'CASH'),
              );
              return Container(
                width: double.infinity,
                margin: const EdgeInsets.only(bottom: 6),
                padding: const EdgeInsets.all(9),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: .72),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: tone.withValues(alpha: .18)),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: 7,
                      height: 7,
                      margin: const EdgeInsets.only(top: 5),
                      decoration:
                          BoxDecoration(color: tone, shape: BoxShape.circle),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            movement.concept.trim().isEmpty
                                ? 'Sin concepto registrado'
                                : movement.concept.trim(),
                            style: const TextStyle(
                              color: Color(0xFF344054),
                              fontSize: 11.5,
                              fontWeight: FontWeight.w900,
                            ),
                          ),
                          if (professional != null &&
                              professional.isNotEmpty) ...[
                            const SizedBox(height: 2),
                            Text(
                              'Profesional: $professional',
                              style: const TextStyle(
                                color: Color(0xFF667085),
                                fontSize: 10.5,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ],
                          if (note != null && note.isNotEmpty) ...[
                            const SizedBox(height: 2),
                            Text(note,
                                style: const TextStyle(
                                  color: Color(0xFF667085),
                                  fontSize: 10.5,
                                  fontStyle: FontStyle.italic,
                                )),
                          ],
                          const SizedBox(height: 3),
                          Text(method,
                              style: TextStyle(
                                color: tone,
                                fontSize: 10,
                                fontWeight: FontWeight.w900,
                              )),
                        ],
                      ),
                    ),
                    const SizedBox(width: 7),
                    Text(
                      '- ${money(movement.amount)}',
                      style: TextStyle(
                        color: tone,
                        fontSize: 11.5,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                  ],
                ),
              );
            }),
          ],
        ],
      ),
    );
  }
}

class _SaleDetailCard extends StatelessWidget {
  final CashSaleModel sale;
  final String Function(double) money;
  final String Function(String?) dateText;
  final String customerName;
  final String barberName;
  final VoidCallback onEdit;
  final VoidCallback onDelete;

  const _SaleDetailCard({
    required this.sale,
    required this.money,
    required this.dateText,
    required this.customerName,
    required this.barberName,
    required this.onEdit,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFFCFBF8),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFF0E7D3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      customerName,
                      style: const TextStyle(
                        fontSize: 14.5,
                        fontWeight: FontWeight.w900,
                        color: Color(0xFF17131E),
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Profesional: $barberName',
                      style: const TextStyle(
                        fontSize: 12.5,
                        fontWeight: FontWeight.w700,
                        color: Color(0xFF667085),
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      dateText(sale.fechaCreacion),
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: Color(0xFF8B7A4E),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    money(sale.total),
                    style: const TextStyle(
                      fontSize: 15.5,
                      fontWeight: FontWeight.w900,
                      color: Color(0xFF16A34A),
                    ),
                  ),
                  const SizedBox(height: 4),
                  _PaymentBreakdownMini(
                    sale: sale,
                    money: money,
                  ),
                  const SizedBox(height: 6),
                  _CompactHistoryActions(onEdit: onEdit, onDelete: onDelete),
                ],
              ),
            ],
          ),
          const SizedBox(height: 10),
          ElectronicDocumentCard(saleId: sale.saleId, showEmpty: false),
          const SizedBox(height: 10),
          _DetailRow('Subtotal', money(sale.subtotal)),
          if (sale.discount > 0) _DetailRow('Descuento', money(sale.discount)),
          if (sale.cashReceived > 0)
            _DetailRow('Recibido', money(sale.cashReceived)),
          if (sale.changeAmount > 0)
            _DetailRow('Vuelto', money(sale.changeAmount)),
          const SizedBox(height: 4),
          if (sale.items.isEmpty)
            const _DetailRow('Items', 'Sin detalle de servicios/productos')
          else
            ...sale.items.map((item) {
              final name = (item.serviceName?.trim().isNotEmpty == true)
                  ? item.serviceName!.trim()
                  : (item.productName?.trim().isNotEmpty == true)
                      ? item.productName!.trim()
                      : 'Item';
              final barber = item.barberUserName?.trim();
              return Padding(
                padding: const EdgeInsets.only(top: 7),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            '$name x${item.cantidad}',
                            style: const TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w900,
                              color: Color(0xFF17131E),
                            ),
                          ),
                        ),
                        Text(
                          money(item.subtotal),
                          style: const TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w900,
                            color: Color(0xFF17131E),
                          ),
                        ),
                      ],
                    ),
                    if (barber != null && barber.isNotEmpty) ...[
                      const SizedBox(height: 2),
                      Text(
                        'Atendido por: $barber',
                        style: const TextStyle(
                          fontSize: 11.8,
                          fontWeight: FontWeight.w700,
                          color: Color(0xFF667085),
                        ),
                      ),
                    ],
                  ],
                ),
              );
            }),
        ],
      ),
    );
  }
}

class _PaymentBreakdownMini extends StatelessWidget {
  final CashSaleModel sale;
  final String Function(double) money;

  const _PaymentBreakdownMini({
    required this.sale,
    required this.money,
  });

  String _label(String code) {
    switch (code.trim().toUpperCase()) {
      case 'CASH':
      case 'EFECTIVO':
        return 'Efectivo';
      case 'YAPE':
        return 'Yape';
      case 'PLIN':
        return 'Plin';
      case 'TRANSFER':
      case 'TRANSFERENCIA':
        return 'Transferencia';
      case 'CARD':
      case 'TARJETA':
        return 'Tarjeta';
      case 'FREE':
      case 'GRATIS':
        return 'Gratis';
      default:
        final clean = code.trim();
        if (clean.isEmpty) return 'Método';
        return clean[0].toUpperCase() + clean.substring(1).toLowerCase();
    }
  }

  @override
  Widget build(BuildContext context) {
    final payments = sale.payments;

    if (payments.isEmpty) {
      return Text(
        sale.metodoPago.toUpperCase(),
        textAlign: TextAlign.right,
        style: const TextStyle(
          fontSize: 11.5,
          fontWeight: FontWeight.w900,
          color: Color(0xFF475467),
        ),
      );
    }

    return Container(
      constraints: const BoxConstraints(maxWidth: 130),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text(
            payments.length > 1
                ? 'MIXTO'
                : _label(payments.first.method).toUpperCase(),
            textAlign: TextAlign.right,
            style: const TextStyle(
              fontSize: 11.5,
              fontWeight: FontWeight.w900,
              color: Color(0xFF475467),
            ),
          ),
          const SizedBox(height: 3),
          ...payments.map(
            (p) => Padding(
              padding: const EdgeInsets.only(top: 2),
              child: Text(
                '${_label(p.method)}: ${money(p.amount)}',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                textAlign: TextAlign.right,
                style: const TextStyle(
                  fontSize: 10.8,
                  fontWeight: FontWeight.w800,
                  color: Color(0xFF667085),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _RangeCashSummaryCard extends StatelessWidget {
  final Map<String, double> payments;
  final double salesTotal;
  final double incomeTotal;
  final double expenseTotal;
  final double cashExpenseTotal;
  final double otherExpenseTotal;
  final double netTotal;
  final double cashExpectedTotal;
  final String Function(double) money;

  const _RangeCashSummaryCard({
    required this.payments,
    required this.salesTotal,
    required this.incomeTotal,
    required this.expenseTotal,
    required this.cashExpenseTotal,
    required this.otherExpenseTotal,
    required this.netTotal,
    required this.cashExpectedTotal,
    required this.money,
  });

  String _label(String code) {
    switch (code.trim().toUpperCase()) {
      case 'CASH':
      case 'EFECTIVO':
        return 'Efectivo';
      case 'YAPE':
        return 'Yape';
      case 'PLIN':
        return 'Plin';
      case 'TRANSFER':
      case 'TRANSFERENCIA':
        return 'Transferencia';
      case 'CARD':
      case 'TARJETA':
        return 'Tarjeta';
      case 'FREE':
      case 'GRATIS':
        return 'Gratis';
      case 'NEQUI':
        return 'Nequi';
      case 'DAVIPLATA':
        return 'Daviplata';
      case 'PAGO_MOVIL':
        return 'Pago móvil';
      case 'ZELLE':
        return 'Zelle';
      case 'QR':
        return 'QR';
      default:
        final clean = code.trim();
        if (clean.isEmpty) return 'Otro';
        return clean[0].toUpperCase() + clean.substring(1).toLowerCase();
    }
  }

  IconData _icon(String code) {
    switch (code.trim().toUpperCase()) {
      case 'CASH':
      case 'EFECTIVO':
        return Icons.payments_rounded;
      case 'YAPE':
      case 'PLIN':
      case 'QR':
        return Icons.qr_code_2_rounded;
      case 'TRANSFER':
      case 'TRANSFERENCIA':
        return Icons.account_balance_rounded;
      case 'CARD':
      case 'TARJETA':
        return Icons.credit_card_rounded;
      case 'FREE':
      case 'GRATIS':
        return Icons.card_giftcard_rounded;
      default:
        return Icons.account_balance_wallet_rounded;
    }
  }

  int _paymentPriority(String code) {
    switch (code.trim().toUpperCase()) {
      case 'CASH':
      case 'EFECTIVO':
        return 0;
      case 'YAPE':
        return 1;
      case 'PLIN':
        return 2;
      case 'TRANSFER':
      case 'TRANSFERENCIA':
        return 3;
      case 'CARD':
      case 'TARJETA':
        return 4;
      default:
        return 99;
    }
  }

  @override
  Widget build(BuildContext context) {
    final visiblePayments = payments.entries.where((e) => e.value > 0).toList()
      ..sort((a, b) {
        final priority =
            _paymentPriority(a.key).compareTo(_paymentPriority(b.key));
        if (priority != 0) return priority;
        return a.key.compareTo(b.key);
      });

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: const Color(0xFFEBDFAF)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 12,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Saldo real del rango',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w900,
              color: Color(0xFF17131E),
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            'Suma saldos reales por método: ventas, ingresos, salidas y traslados.',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: Color(0xFF7A7282),
            ),
          ),
          const SizedBox(height: 14),
          if (visiblePayments.isNotEmpty) ...[
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: visiblePayments.map((entry) {
                return Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF8F4EA),
                    borderRadius: BorderRadius.circular(999),
                    border: Border.all(color: const Color(0xFFEBDFAF)),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        _icon(entry.key),
                        size: 15,
                        color: const Color(0xFF7A5A22),
                      ),
                      const SizedBox(width: 6),
                      Text(
                        '${_label(entry.key)}: ${money(entry.value)}',
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w900,
                          color: Color(0xFF7A5A22),
                        ),
                      ),
                    ],
                  ),
                );
              }).toList(),
            ),
            const SizedBox(height: 14),
          ],
          _SummaryRow('Ventas registradas', money(salesTotal)),
          _SummaryRow('Ingresos adicionales', money(incomeTotal)),
          _SummaryRow('Salidas totales', '- ${money(expenseTotal)}'),
          _SummaryRow(
            'De ellas, salidas en efectivo',
            '- ${money(cashExpenseTotal)}',
          ),
          _SummaryRow(
            'Salidas por otros métodos',
            '- ${money(otherExpenseTotal)}',
          ),
          const Divider(height: 22),
          _SummaryRow(
            'Total neto registrado',
            money(netTotal),
            strong: true,
          ),
          const SizedBox(height: 6),
          _SummaryRow(
            'Efectivo físico esperado',
            money(cashExpectedTotal),
            strong: true,
            gold: true,
          ),
        ],
      ),
    );
  }
}

class _SummaryRow extends StatelessWidget {
  final String label;
  final String value;
  final bool strong;
  final bool gold;

  const _SummaryRow(
    this.label,
    this.value, {
    this.strong = false,
    this.gold = false,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 7),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontSize: strong ? 13.5 : 12.5,
                fontWeight: strong ? FontWeight.w900 : FontWeight.w700,
                color: const Color(0xFF667085),
              ),
            ),
          ),
          const SizedBox(width: 10),
          Flexible(
            flex: 0,
            child: FittedBox(
              fit: BoxFit.scaleDown,
              child: Text(
                value,
                maxLines: 1,
                textAlign: TextAlign.right,
                style: TextStyle(
                  fontSize: strong ? 14 : 13,
                  fontWeight: FontWeight.w900,
                  color:
                      gold ? const Color(0xFF7A5A22) : const Color(0xFF17131E),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _PaymentMethodsWrap extends StatelessWidget {
  final List<PaymentMethodSummaryModel> payments;
  final String Function(double) money;
  final bool compact;

  const _PaymentMethodsWrap({
    required this.payments,
    required this.money,
    required this.compact,
  });

  String _label(String code) {
    switch (code.trim().toUpperCase()) {
      case 'CASH':
      case 'EFECTIVO':
        return 'Efectivo';
      case 'YAPE':
        return 'Yape';
      case 'PLIN':
        return 'Plin';
      case 'TRANSFER':
      case 'TRANSFERENCIA':
        return 'Transferencia';
      case 'CARD':
      case 'TARJETA':
        return 'Tarjeta';
      case 'FREE':
      case 'GRATIS':
        return 'Gratis';
      case 'NEQUI':
        return 'Nequi';
      case 'DAVIPLATA':
        return 'Daviplata';
      case 'PAGO_MOVIL':
        return 'Pago móvil';
      case 'ZELLE':
        return 'Zelle';
      case 'QR':
        return 'QR';
      default:
        final clean = code.trim();
        if (clean.isEmpty) return 'Otro';
        return clean[0].toUpperCase() + clean.substring(1).toLowerCase();
    }
  }

  IconData _icon(String code) {
    switch (code.trim().toUpperCase()) {
      case 'CASH':
      case 'EFECTIVO':
        return Icons.payments_rounded;
      case 'YAPE':
      case 'PLIN':
      case 'QR':
        return Icons.qr_code_2_rounded;
      case 'TRANSFER':
      case 'TRANSFERENCIA':
        return Icons.account_balance_rounded;
      case 'CARD':
      case 'TARJETA':
        return Icons.credit_card_rounded;
      case 'FREE':
      case 'GRATIS':
        return Icons.card_giftcard_rounded;
      default:
        return Icons.account_balance_wallet_rounded;
    }
  }

  Color _softColor(String code) {
    switch (code.trim().toUpperCase()) {
      case 'CASH':
      case 'EFECTIVO':
        return const Color(0xFFEAF7EE);
      case 'YAPE':
        return const Color(0xFFF1EAFE);
      case 'PLIN':
        return const Color(0xFFEAF2FF);
      case 'TRANSFER':
      case 'TRANSFERENCIA':
        return const Color(0xFFEEF2F6);
      case 'CARD':
      case 'TARJETA':
        return const Color(0xFFF3F4F6);
      default:
        return const Color(0xFFF8F4EA);
    }
  }

  Color _textColor(String code) {
    switch (code.trim().toUpperCase()) {
      case 'CASH':
      case 'EFECTIVO':
        return const Color(0xFF166534);
      case 'YAPE':
        return const Color(0xFF6D28D9);
      case 'PLIN':
        return const Color(0xFF1D4ED8);
      case 'TRANSFER':
      case 'TRANSFERENCIA':
        return const Color(0xFF334155);
      case 'CARD':
      case 'TARJETA':
        return const Color(0xFF111827);
      default:
        return const Color(0xFF7A5A22);
    }
  }

  @override
  Widget build(BuildContext context) {
    final visible = payments.where((p) => p.totalAmount > 0).toList();

    if (visible.isEmpty) {
      return const SizedBox.shrink();
    }

    if (!compact) {
      return Column(
        children: visible
            .map(
              (p) => _PaymentMethodRow(
                icon: _icon(p.paymentMethod),
                label: _label(p.paymentMethod),
                count: p.count,
                amount: money(p.totalAmount),
                bg: _softColor(p.paymentMethod),
                fg: _textColor(p.paymentMethod),
              ),
            )
            .toList(),
      );
    }

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: visible
          .take(4)
          .map(
            (p) => Container(
              padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6),
              decoration: BoxDecoration(
                color: _softColor(p.paymentMethod),
                borderRadius: BorderRadius.circular(999),
                border: Border.all(
                  color: _textColor(p.paymentMethod).withOpacity(0.14),
                ),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    _icon(p.paymentMethod),
                    size: 14,
                    color: _textColor(p.paymentMethod),
                  ),
                  const SizedBox(width: 5),
                  ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 170),
                    child: Text(
                      '${_label(p.paymentMethod)}: ${money(p.totalAmount)}',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w900,
                        color: _textColor(p.paymentMethod),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          )
          .toList(),
    );
  }
}

class _PaymentMethodRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final int count;
  final String amount;
  final Color bg;
  final Color fg;

  const _PaymentMethodRow({
    required this.icon,
    required this.label,
    required this.count,
    required this.amount,
    required this.bg,
    required this.fg,
  });

  @override
  Widget build(BuildContext context) {
    final title = count > 0 ? '$label · $count venta(s)' : label;

    return Container(
      margin: const EdgeInsets.only(bottom: 9),
      padding: const EdgeInsets.all(11),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: fg.withOpacity(0.12)),
      ),
      child: Row(
        children: [
          Container(
            width: 32,
            height: 32,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.75),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, size: 17, color: fg),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              title,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontSize: 12.5,
                fontWeight: FontWeight.w900,
                color: fg,
              ),
            ),
          ),
          const SizedBox(width: 6),
          Flexible(
            flex: 0,
            child: FittedBox(
              fit: BoxFit.scaleDown,
              child: Text(
                amount,
                maxLines: 1,
                textAlign: TextAlign.right,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w900,
                  color: fg,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MovementDetailCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final String amount;
  final String date;
  final String? barberName;
  final String? note;
  final bool isIncome;
  final VoidCallback onEdit;
  final VoidCallback onDelete;

  const _MovementDetailCard({
    required this.title,
    required this.subtitle,
    required this.amount,
    required this.date,
    this.barberName,
    this.note,
    this.isIncome = false,
    required this.onEdit,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    final cleanBarber = barberName?.trim();
    final cleanNote = note?.trim();

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFFCFBF8),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFF0E7D3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w900,
                    color: Color(0xFF17131E),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    amount,
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w900,
                      color: isIncome
                          ? const Color(0xFF16A34A)
                          : const Color(0xFFC2410C),
                    ),
                  ),
                  const SizedBox(height: 6),
                  _CompactHistoryActions(onEdit: onEdit, onDelete: onDelete),
                ],
              ),
            ],
          ),
          const SizedBox(height: 5),
          Text(
            subtitle,
            style: const TextStyle(
              fontSize: 12.2,
              fontWeight: FontWeight.w800,
              color: Color(0xFF667085),
            ),
          ),
          const SizedBox(height: 3),
          Text(
            date,
            style: const TextStyle(
              fontSize: 11.8,
              fontWeight: FontWeight.w700,
              color: Color(0xFF8B7A4E),
            ),
          ),
          if (cleanBarber != null && cleanBarber.isNotEmpty) ...[
            const SizedBox(height: 3),
            Text(
              'Profesional: $cleanBarber',
              style: const TextStyle(
                fontSize: 11.8,
                fontWeight: FontWeight.w700,
                color: Color(0xFF667085),
              ),
            ),
          ],
          if (cleanNote != null && cleanNote.isNotEmpty) ...[
            const SizedBox(height: 6),
            Text(
              cleanNote,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: Color(0xFF475467),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _CompactHistoryActions extends StatelessWidget {
  final VoidCallback onEdit;
  final VoidCallback onDelete;

  const _CompactHistoryActions({
    required this.onEdit,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        InkWell(
          onTap: onEdit,
          borderRadius: BorderRadius.circular(18),
          child: const Padding(
            padding: EdgeInsets.all(5),
            child:
                Icon(Icons.edit_outlined, size: 17, color: Color(0xFF344054)),
          ),
        ),
        const SizedBox(width: 2),
        InkWell(
          onTap: onDelete,
          borderRadius: BorderRadius.circular(18),
          child: const Padding(
            padding: EdgeInsets.all(5),
            child:
                Icon(Icons.delete_outline, size: 17, color: Color(0xFFB42318)),
          ),
        ),
      ],
    );
  }
}

class _AuditLogCard extends StatelessWidget {
  final CashAuditLogModel log;
  final String Function(String?) dateText;

  const _AuditLogCard({
    required this.log,
    required this.dateText,
  });

  String get _actionLabel {
    switch (log.action.toUpperCase()) {
      case 'UPDATE':
        return 'Edito';
      case 'DELETE':
        return 'Elimino';
      default:
        return log.action.isEmpty ? 'Cambio' : log.action;
    }
  }

  String get _entityLabel {
    switch (log.entityType.toUpperCase()) {
      case 'SALE':
        return 'venta';
      case 'CASH_MOVEMENT':
        return 'movimiento';
      default:
        return 'registro';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: const Color(0xFFFFFBF3),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFEED9A8)),
      ),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          tilePadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 2),
          childrenPadding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
          title: Text(
            '$_actionLabel $_entityLabel #${log.entityId ?? '-'}',
            style: const TextStyle(
              fontSize: 13.5,
              fontWeight: FontWeight.w900,
              color: Color(0xFF17131E),
            ),
          ),
          subtitle: Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              '${dateText(log.createdAt)} - ${log.actorUserName ?? 'Sistema'}',
              style: const TextStyle(
                fontSize: 11.8,
                fontWeight: FontWeight.w700,
                color: Color(0xFF667085),
              ),
            ),
          ),
          children: [
            if (log.reason?.trim().isNotEmpty == true)
              _DetailRow('Motivo', log.reason!.trim()),
            _AuditSnapshot(label: 'Antes', value: log.beforeSnapshot),
            _AuditSnapshot(label: 'Despues', value: log.afterSnapshot),
          ],
        ),
      ),
    );
  }
}

class _AuditSnapshot extends StatelessWidget {
  final String label;
  final String? value;

  const _AuditSnapshot({
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    final text = value?.trim().isNotEmpty == true ? value!.trim() : 'Sin datos';
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(top: 8),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xFF17131E),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w900,
              color: Color(0xFFD6B36A),
            ),
          ),
          const SizedBox(height: 6),
          Text(
            text,
            maxLines: 8,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }
}

class _InlineError extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;
  final String retryLabel;

  const _InlineError({
    required this.message,
    required this.onRetry,
    this.retryLabel = 'Reintentar ventas',
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFFFFBFA),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFFDA29B)),
      ),
      child: Column(
        children: [
          Text(
            message,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 12.5,
              fontWeight: FontWeight.w700,
              color: Color(0xFFB42318),
            ),
          ),
          const SizedBox(height: 8),
          OutlinedButton(
            onPressed: onRetry,
            child: Text(retryLabel),
          ),
        ],
      ),
    );
  }
}

class _DetailBlock extends StatelessWidget {
  final String title;
  final List<Widget> children;

  const _DetailBlock({
    required this.title,
    required this.children,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFFF0E7D3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 15.5,
              fontWeight: FontWeight.w900,
              color: Color(0xFF17131E),
            ),
          ),
          const SizedBox(height: 10),
          ...children,
        ],
      ),
    );
  }
}

class _DetailRow extends StatelessWidget {
  final String label;
  final String value;

  const _DetailRow(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 9),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: const TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w700,
                color: Color(0xFF667085),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Flexible(
            child: Text(
              value,
              textAlign: TextAlign.right,
              style: const TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w900,
                color: Color(0xFF17131E),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MessageCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String message;
  final VoidCallback? onRetry;

  const _MessageCard({
    required this.icon,
    required this.title,
    required this.message,
    this.onRetry,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: const Color(0xFFF0E7D3)),
      ),
      child: Column(
        children: [
          Icon(icon, size: 36, color: const Color(0xFF98A2B3)),
          const SizedBox(height: 10),
          Text(
            title,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w900,
              color: Color(0xFF17131E),
            ),
          ),
          const SizedBox(height: 6),
          Text(
            message,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: Color(0xFF667085),
            ),
          ),
          if (onRetry != null) ...[
            const SizedBox(height: 14),
            OutlinedButton(
              onPressed: onRetry,
              child: const Text('Reintentar'),
            ),
          ],
        ],
      ),
    );
  }
}
