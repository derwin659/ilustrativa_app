import 'dart:async';

import 'package:flutter/material.dart';
import '../../l10n/owner_core_copy.dart';
import '../../constants/admin_permission_keys.dart';
import '../../model/owner_cash_models.dart';
import '../../model/owner_product_models.dart';
import '../../services/owner_cash_service.dart';
import '../../services/owner_product_service.dart';
import '../../services/session_service.dart';
import '../../services/tenant_money_service.dart';
import '../../widgets/super_gods_ui.dart';
import '../../widgets/electronic_document_card.dart';
import 'business_labels.dart';
import 'owner_barber_payment_screen.dart';
import 'owner_cash_sale_create_screen.dart';
import 'owner_close_cash_screen.dart';
import 'owner_open_cash_screen.dart';
import 'owner_cash_movement_sheet.dart';
import 'owner_cash_reconciliation_sheet.dart';
import 'owner_cash_fund_sheet.dart';
import 'owner_cash_history_screen.dart';
import 'owner_pending_payments_screen.dart';

BusinessLabels _ownerLabels = BusinessLabels.fromType('BARBERSHOP');

class OwnerCashScreen extends StatefulWidget {
  const OwnerCashScreen({super.key});

  @override
  State<OwnerCashScreen> createState() => _OwnerCashScreenState();
}

class _OwnerCashScreenState extends State<OwnerCashScreen> {
  final OwnerCashService _service = OwnerCashService();
  final OwnerProductService _productService = OwnerProductService();

  List<BranchOptionModel> _branches = [];
  int? _selectedBranchId;

  bool _loading = true;
  bool _navigating = false;
  bool _silentRefreshing = false;
  String? _error;
  CashRegisterModel? _cash;
  CashRegisterModel? _pendingReconciliation;
  List<CashSaleModel> _sales = [];
  List<CashSaleModel> _pendingValidationSales = [];
  List<OwnerProductOrderModel> _productOrders = [];
  List<OwnerLocalConsumptionOrderModel> _localConsumptionOrders = [];
  int? _processingProductOrderId;
  int? _processingLocalConsumptionOrderId;
  List<OwnerPaymentMethodModel> _paymentMethods =
      OwnerPaymentMethodModel.defaultPeruFallback();

  String _sessionRole = '';
  bool _canManageFund = false;
  Timer? _autoRefreshTimer;
  DateTime? _lastUpdatedAt;

  bool get _canManageSales => _sessionRole.trim().toUpperCase() == 'OWNER';

  String _normalizePaymentMethod(String method) => normalizePaymentCode(method);

  double _sumSalesByPayment(Set<String> methods) {
    return _sales.fold<double>(0, (sum, sale) {
      final method = _normalizePaymentMethod(sale.metodoPago);
      return methods.contains(method) ? sum + sale.total : sum;
    });
  }

  int _countSalesByPayment(Set<String> methods) {
    return _sales.where((sale) {
      final method = _normalizePaymentMethod(sale.metodoPago);
      return methods.contains(method);
    }).length;
  }

  double get _yapeTotal => _sumSalesByPayment({'YAPE'});
  double get _plinTotal => _sumSalesByPayment({'PLIN'});
  double get _cardTotal =>
      _sumSalesByPayment({'CARD', 'TARJETA', 'VISA', 'MASTERCARD'});

  int get _yapeCount => _countSalesByPayment({'YAPE'});
  int get _plinCount => _countSalesByPayment({'PLIN'});
  int get _cardCount =>
      _countSalesByPayment({'CARD', 'TARJETA', 'VISA', 'MASTERCARD'});

  bool _isCourtesySale(CashSaleModel sale) {
    return _normalizePaymentMethod(sale.metodoPago) == 'FREE';
  }

  double _saleReferenceValue(CashSaleModel sale) {
    if (sale.subtotal > 0) return sale.subtotal;

    final itemsTotal = sale.items.fold<double>(
      0,
      (sum, item) => sum + item.subtotal,
    );

    return itemsTotal > 0 ? itemsTotal : sale.discount;
  }

  String _readSaleBarberName(CashSaleModel sale) {
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

    return '${_ownerLabels.professionalSingular} no registrado';
  }

  List<_CourtesyEntry> get _courtesyEntries {
    final entries = <_CourtesyEntry>[];

    for (final sale in _sales.where(_isCourtesySale)) {
      final serviceItems = sale.items.where((item) {
        final hasService = item.serviceId != null ||
            (item.serviceName != null && item.serviceName!.trim().isNotEmpty);
        final hasProduct = item.productId != null ||
            (item.productName != null && item.productName!.trim().isNotEmpty);
        return hasService || !hasProduct;
      }).toList();

      if (serviceItems.isEmpty) {
        entries.add(
          _CourtesyEntry(
            barberName: _readSaleBarberName(sale),
            count: 1,
            referenceValue: _saleReferenceValue(sale),
          ),
        );
        continue;
      }

      for (final item in serviceItems) {
        final itemBarber = item.barberUserName?.trim();
        final saleBarber = sale.barberName?.trim();
        final barberName = itemBarber != null && itemBarber.isNotEmpty
            ? itemBarber
            : saleBarber != null && saleBarber.isNotEmpty
                ? saleBarber
                : '${_ownerLabels.professionalSingular} no registrado';

        entries.add(
          _CourtesyEntry(
            barberName: barberName,
            count: item.cantidad <= 0 ? 1 : item.cantidad,
            referenceValue:
                item.subtotal > 0 ? item.subtotal : item.precioUnitario,
          ),
        );
      }
    }

    return entries;
  }

  int get _courtesyCutsCount {
    return _courtesyEntries.fold<int>(0, (sum, entry) => sum + entry.count);
  }

  double get _courtesyReferenceValue {
    return _courtesyEntries.fold<double>(
      0,
      (sum, entry) => sum + entry.referenceValue,
    );
  }

  List<_BarberCourtesySummary> get _courtesyByBarber {
    final map = <String, _BarberCourtesySummary>{};

    for (final entry in _courtesyEntries) {
      final current = map[entry.barberName];
      if (current == null) {
        map[entry.barberName] = _BarberCourtesySummary(
          barberName: entry.barberName,
          count: entry.count,
          referenceValue: entry.referenceValue,
        );
      } else {
        map[entry.barberName] = current.copyWith(
          count: current.count + entry.count,
          referenceValue: current.referenceValue + entry.referenceValue,
        );
      }
    }

    final result = map.values.toList()
      ..sort((a, b) {
        final byCount = b.count.compareTo(a.count);
        if (byCount != 0) return byCount;
        return b.referenceValue.compareTo(a.referenceValue);
      });

    return result;
  }

  Future<void> _loadBusinessLabels() async {
    final businessType = await SessionService.getBusinessType();
    if (!mounted) return;

    setState(() {
      _ownerLabels = BusinessLabels.fromType(businessType);
    });

    debugPrint(
        '🏷️ ${runtimeType} businessType => $businessType · ${_ownerLabels.professionalsPlural}');
  }

  @override
  void initState() {
    super.initState();
    _loadBusinessLabels();
    _boot();

    _autoRefreshTimer = Timer.periodic(
      const Duration(seconds: 10),
      (_) => _silentRefresh(),
    );
  }

  @override
  void dispose() {
    _autoRefreshTimer?.cancel();
    super.dispose();
  }

  Future<void> _boot() async {
    final role = await SessionService.getRole();
    final canManageFund = await SessionService.hasPermission(
      AdminPermissionKeys.cashFundManage,
    );
    debugPrint('SESSION ROLE => $role');

    if (!mounted) return;
    setState(() {
      _sessionRole = (role ?? '').trim().toUpperCase();
      _canManageFund = canManageFund;
    });
    await _initialize();
  }

  String get _selectedBranchName {
    final branchId = _selectedBranchId;
    if (branchId == null) return 'Sede';

    for (final b in _branches) {
      if (b.id == branchId) {
        final name = b.name.trim();
        return name.isEmpty ? 'Sede ${b.id}' : name;
      }
    }
    return 'Sede';
  }

  String _cleanError(Object e) => SuperGodsUi.humanError(e);

  bool _isNoOpenCashMessage(String message) {
    final text = message.toLowerCase();
    return text.contains('no hay una caja abierta en esta sede');
  }

  bool _isCashAlreadyClosedMessage(String message) {
    final text = message.toLowerCase();
    return text.contains('la caja ya no está abierta');
  }

  Future<void> _initialize() async {
    if (!mounted) return;

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final branches = await _service.getOwnerBranches();
      final sessionBranchId = await SessionService.getBranchId();

      if (!mounted) return;

      if (branches.isEmpty) {
        setState(() {
          _branches = [];
          _selectedBranchId = null;
          _cash = null;
          _sales = [];
          _pendingValidationSales = [];
          _localConsumptionOrders = [];
          _error = 'No tienes sedes registradas para este negocio.';
          _loading = false;
        });
        return;
      }

      final selected = _selectedBranchId != null &&
              branches.any((b) => b.id == _selectedBranchId)
          ? _selectedBranchId!
          : sessionBranchId != null &&
                  branches.any((b) => b.id == sessionBranchId)
              ? sessionBranchId
              : branches.first.id;

      final selectedBranch = branches.firstWhere(
        (b) => b.id == selected,
        orElse: () => branches.first,
      );

      await SessionService.saveBranch(
        branchId: selectedBranch.id,
        branchName: selectedBranch.name,
      );

      if (!mounted) return;

      setState(() {
        _branches = branches;
        _selectedBranchId = selectedBranch.id;
      });

      await _load();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _branches = [];
        _selectedBranchId = null;
        _cash = null;
        _sales = [];
        _pendingValidationSales = [];
        _error = _cleanError(e);
        _loading = false;
      });
    }
  }

  Future<void> _load() async {
    final branchId = _selectedBranchId;
    if (!mounted || branchId == null) return;

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final results = await Future.wait([
        _service.getCurrentCashRegister(branchId),
        _service.getTodaySales(branchId),
        _service.getOwnerPaymentMethods(branchId),
        _loadProductOrdersForBranch(branchId),
        _loadLocalConsumptionOrdersForBranch(branchId),
        _service.getPendingValidationSales(branchId),
      ]);

      final cash = results[0] as CashRegisterModel?;
      final sales = results[1] as List<CashSaleModel>;
      final methods = results[2] as List<OwnerPaymentMethodModel>;
      final productOrders = results[3] as List<OwnerProductOrderModel>;
      final localConsumptionOrders =
          results[4] as List<OwnerLocalConsumptionOrderModel>;
      final pendingValidationSales = results[5] as List<CashSaleModel>;
      final pendingReconciliation =
          await _service.getPendingCashReconciliation(branchId);

      if (!mounted) return;
      setState(() {
        _cash = cash;
        _pendingReconciliation = pendingReconciliation;
        _sales = sales;
        _productOrders = productOrders;
        _localConsumptionOrders = localConsumptionOrders;
        _pendingValidationSales = pendingValidationSales;
        _paymentMethods = methods.isEmpty
            ? OwnerPaymentMethodModel.defaultPeruFallback()
            : methods;
        _error = null;
        _loading = false;
        _lastUpdatedAt = DateTime.now();
      });
    } catch (e) {
      final message = _cleanError(e);

      if (!mounted) return;

      if (_isNoOpenCashMessage(message) ||
          _isCashAlreadyClosedMessage(message)) {
        final productOrders = await _loadProductOrdersForBranch(branchId)
            .catchError((_) => <OwnerProductOrderModel>[]);
        final localConsumptionOrders =
            await _loadLocalConsumptionOrdersForBranch(branchId)
                .catchError((_) => <OwnerLocalConsumptionOrderModel>[]);
        final pendingReconciliation =
            await _service.getPendingCashReconciliation(branchId);

        if (!mounted) return;
        setState(() {
          _cash = null;
          _pendingReconciliation = pendingReconciliation;
          _sales = [];
          _productOrders = productOrders;
          _localConsumptionOrders = localConsumptionOrders;
          _pendingValidationSales = [];
          _error = null;
          _loading = false;
          _lastUpdatedAt = DateTime.now();
        });
        return;
      }

      setState(() {
        _cash = null;
        _pendingReconciliation = null;
        _sales = [];
        _pendingValidationSales = [];
        _productOrders = [];
        _localConsumptionOrders = [];
        _error = message;
        _loading = false;
      });
    }
  }

  Future<void> _silentRefresh() async {
    if (!mounted || _loading || _navigating || _silentRefreshing) return;

    final branchId = _selectedBranchId;
    if (branchId == null) return;

    _silentRefreshing = true;

    try {
      final results = await Future.wait([
        _service.getCurrentCashRegister(branchId),
        _service.getTodaySales(branchId),
        _service.getOwnerPaymentMethods(branchId),
        _loadProductOrdersForBranch(branchId),
        _loadLocalConsumptionOrdersForBranch(branchId),
        _service.getPendingValidationSales(branchId),
      ]);

      final cash = results[0] as CashRegisterModel?;
      final sales = results[1] as List<CashSaleModel>;
      final methods = results[2] as List<OwnerPaymentMethodModel>;
      final productOrders = results[3] as List<OwnerProductOrderModel>;
      final localConsumptionOrders =
          results[4] as List<OwnerLocalConsumptionOrderModel>;
      final pendingValidationSales = results[5] as List<CashSaleModel>;

      if (!mounted) return;

      setState(() {
        _cash = cash;
        _sales = sales;
        _productOrders = productOrders;
        _localConsumptionOrders = localConsumptionOrders;
        _pendingValidationSales = pendingValidationSales;
        _paymentMethods = methods.isEmpty
            ? OwnerPaymentMethodModel.defaultPeruFallback()
            : methods;
        _error = null;
        _lastUpdatedAt = DateTime.now();
      });
    } catch (e) {
      final message = _cleanError(e);

      if (!mounted) return;

      if (_isNoOpenCashMessage(message) ||
          _isCashAlreadyClosedMessage(message)) {
        final productOrders = await _loadProductOrdersForBranch(branchId)
            .catchError((_) => <OwnerProductOrderModel>[]);
        final localConsumptionOrders =
            await _loadLocalConsumptionOrdersForBranch(branchId)
                .catchError((_) => <OwnerLocalConsumptionOrderModel>[]);

        if (!mounted) return;
        setState(() {
          _cash = null;
          _sales = [];
          _productOrders = productOrders;
          _localConsumptionOrders = localConsumptionOrders;
          _pendingValidationSales = [];
          _error = null;
          _lastUpdatedAt = DateTime.now();
        });
      }
    } finally {
      _silentRefreshing = false;
    }
  }

  Future<List<OwnerProductOrderModel>> _loadProductOrdersForBranch(
      int branchId) async {
    try {
      final pending = await _productService.getProductOrders(
        branchId: branchId,
        status: 'PENDING',
      );
      final approved = await _productService.getProductOrders(
        branchId: branchId,
        status: 'APPROVED',
      );

      final orders = <OwnerProductOrderModel>[...pending, ...approved];
      orders.sort((a, b) => (b.createdAt ?? '').compareTo(a.createdAt ?? ''));
      return orders;
    } catch (e) {
      debugPrint('OWNER CASH PRODUCT ORDERS ERROR => $e');
      return <OwnerProductOrderModel>[];
    }
  }

  Future<List<OwnerLocalConsumptionOrderModel>>
      _loadLocalConsumptionOrdersForBranch(int branchId) async {
    try {
      final pending = await _productService.getLocalConsumptionOrders(
        branchId: branchId,
        status: 'PENDING',
      );

      pending.sort((a, b) => (b.createdAt ?? '').compareTo(a.createdAt ?? ''));
      return pending;
    } catch (e) {
      debugPrint('OWNER CASH LOCAL CONSUMPTION ORDERS ERROR => $e');
      return <OwnerLocalConsumptionOrderModel>[];
    }
  }

  Future<void> _changeBranch(int value) async {
    if (_selectedBranchId == value || _navigating) return;

    final branch = _branches.firstWhere(
      (b) => b.id == value,
      orElse: () => BranchOptionModel(id: value, name: ''),
    );

    await SessionService.saveBranch(
      branchId: branch.id,
      branchName: branch.name,
    );

    if (!mounted) return;

    setState(() {
      _selectedBranchId = value;
      _cash = null;
      _sales = [];
      _localConsumptionOrders = [];
      _error = null;
      _lastUpdatedAt = null;
    });

    await _load();
  }

  Future<void> _runNavigation(Future<void> Function() action) async {
    if (_navigating) return;

    setState(() {
      _navigating = true;
    });

    try {
      await action();
    } finally {
      if (!mounted) return;
      setState(() {
        _navigating = false;
      });
    }
  }

  Future<void> _goOpenCash() async {
    final branchId = _selectedBranchId;
    if (branchId == null) return;

    await _runNavigation(() async {
      await Navigator.push<bool>(
        context,
        MaterialPageRoute(
          builder: (_) => OwnerOpenCashScreen(
            branchId: branchId,
            branchName: _selectedBranchName,
            canManageFund: _canManageFund,
          ),
        ),
      );

      if (!mounted) return;
      await _load();
    });
  }

  Future<void> _goCloseCash() async {
    final branchId = _selectedBranchId;
    if (branchId == null) return;

    await _runNavigation(() async {
      await _load();

      if (!mounted) return;

      if (_cash == null) {
        _showInfo('La caja ya no está abierta en esta sede.');
        return;
      }

      await Navigator.push<bool>(
        context,
        MaterialPageRoute(
          builder: (_) => OwnerCloseCashScreen(
            cashRegister: _cash!,
            branchId: branchId,
            branchName: _selectedBranchName,
            canManageFund: _canManageFund,
          ),
        ),
      );

      if (!mounted) return;
      await _load();
    });
  }

  Future<void> _goNewSale() async {
    final branchId = _selectedBranchId;
    if (branchId == null) return;

    await _runNavigation(() async {
      await _load();

      if (!mounted) return;

      if (_cash == null) {
        _showInfo('Primero abre una caja en esta sede.');
        return;
      }

      await Navigator.push<bool>(
        context,
        MaterialPageRoute(
          builder: (_) => OwnerCashSaleCreateScreen(
            branchId: branchId,
            branchName: _selectedBranchName,
          ),
        ),
      );

      if (!mounted) return;
      await _load();
    });
  }

  Future<void> _showFundSheet() async {
    if (!_canManageFund) {
      _showInfo(
          'Solo el dueño o un administrador autorizado puede gestionar el fondo.');
      return;
    }
    final branchId = _selectedBranchId;
    if (branchId == null) return;

    final changed = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => OwnerCashFundSheet(
        branchId: branchId,
        branchName: _selectedBranchName,
      ),
    );

    if (changed == true && mounted) {
      _showInfo('Fondo acumulado actualizado correctamente.');
      await _load();
    }
  }

  Future<void> _showMovementSheet({CashMovementModel? movement}) async {
    if (movement?.fundingSource == 'ACCUMULATED_FUND' && !_canManageFund) {
      _showInfo(
          'No tienes permiso para modificar movimientos del fondo acumulado.');
      return;
    }
    final branchId = _selectedBranchId;
    final cash = _cash;
    if (branchId == null || cash == null) {
      _showInfo('Primero abre una caja en esta sede.');
      return;
    }

    final changed = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => OwnerCashMovementSheet(
        branchId: branchId,
        branchName: _selectedBranchName,
        cashRegisterId: cash.id,
        initialMovement: movement,
        canManageFund: _canManageFund,
      ),
    );

    if (changed == true && mounted) {
      _showInfo(movement == null
          ? 'Movimiento registrado correctamente.'
          : 'Movimiento actualizado correctamente.');
      await _load();
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

  Future<void> _handleDeleteMovement(CashMovementModel movement) async {
    if (movement.fundingSource == 'ACCUMULATED_FUND' && !_canManageFund) {
      _showInfo(
          'No tienes permiso para eliminar movimientos del fondo acumulado.');
      return;
    }
    final branchId = _selectedBranchId;
    if (branchId == null) return;

    final ok = await SuperGodsUi.confirm(
      context,
      title: 'Eliminar movimiento',
      message:
          'Esta acción retirará el movimiento y quedará registrada en auditoría. No se puede deshacer.',
      confirmLabel: 'Eliminar',
      icon: Icons.delete_outline_rounded,
      destructive: true,
    );

    if (ok != true) return;

    final auditReason = await _requestAuditReason('Motivo de eliminacion');
    if (auditReason == null) return;

    try {
      await _service.deleteCashMovement(
        branchId: branchId,
        movementId: movement.id,
        auditReason: auditReason,
      );
      if (!mounted) return;
      _showInfo('Movimiento eliminado correctamente.');
      await _load();
    } catch (e) {
      if (!mounted) return;
      _showInfo(_cleanError(e));
    }
  }

  void _showInfo(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        behavior: SnackBarBehavior.floating,
        content: Text(message),
      ),
    );
  }

  Future<void> _handleProductOrderAction(
    OwnerProductOrderModel order,
    String action,
  ) async {
    final branchId = _selectedBranchId;
    if (branchId == null) return;

    if (action == 'deliver' && _cash == null) {
      _showInfo('Abre caja para entregar productos separados.');
      return;
    }

    setState(() => _processingProductOrderId = order.id);

    try {
      if (action == 'approve') {
        await _productService.approveProductOrder(
          branchId: branchId,
          orderId: order.id,
        );
        _showInfo('Producto separado aprobado.');
      } else if (action == 'reject') {
        await _productService.rejectProductOrder(
          branchId: branchId,
          orderId: order.id,
        );
        _showInfo('Pedido de producto rechazado.');
      } else {
        await _productService.deliverProductOrder(
          branchId: branchId,
          orderId: order.id,
        );
        _showInfo('Producto entregado correctamente.');
      }

      await _load();
    } catch (e) {
      if (!mounted) return;
      _showInfo(_cleanError(e));
    } finally {
      if (mounted) setState(() => _processingProductOrderId = null);
    }
  }

  Future<void> _handleLocalConsumptionOrderAction(
    OwnerLocalConsumptionOrderModel order,
    String action,
  ) async {
    final branchId = _selectedBranchId;
    if (branchId == null) return;

    if (action == 'complete' && _cash == null) {
      _showInfo('Abre caja para atender consumos del QR.');
      return;
    }

    String? paymentMethod;

    if (action == 'complete') {
      paymentMethod = await showDialog<String>(
        context: context,
        builder: (_) => _LocalConsumptionPaymentDialog(total: order.total),
      );
      if (paymentMethod == null) return;
    }

    setState(() => _processingLocalConsumptionOrderId = order.id);

    try {
      if (action == 'reject') {
        await _productService.rejectLocalConsumptionOrder(
          branchId: branchId,
          orderId: order.id,
        );
        _showInfo('Consumo del QR rechazado.');
      } else {
        await _productService.completeLocalConsumptionOrder(
          branchId: branchId,
          orderId: order.id,
          paymentMethod: paymentMethod,
        );
        _showInfo('Consumo marcado como atendido.');
      }

      await _load();
    } catch (e) {
      if (!mounted) return;
      _showInfo(_cleanError(e));
    } finally {
      if (mounted) setState(() => _processingLocalConsumptionOrderId = null);
    }
  }

  Future<void> _handleEditSale(CashSaleModel sale) async {
    final branchId = _selectedBranchId;
    if (branchId == null) return;

    final metodoCtrl = TextEditingController(text: sale.metodoPago);
    final subtotalCtrl =
        TextEditingController(text: sale.subtotal.toStringAsFixed(2));
    final discountCtrl =
        TextEditingController(text: sale.discount.toStringAsFixed(2));
    final totalCtrl =
        TextEditingController(text: sale.total.toStringAsFixed(2));
    final cashReceivedCtrl =
        TextEditingController(text: sale.cashReceived.toStringAsFixed(2));
    final changeCtrl =
        TextEditingController(text: sale.changeAmount.toStringAsFixed(2));

    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Editar venta'),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                  controller: metodoCtrl,
                  decoration:
                      const InputDecoration(labelText: 'Método de pago')),
              TextField(
                  controller: subtotalCtrl,
                  decoration: const InputDecoration(labelText: 'Subtotal')),
              TextField(
                  controller: discountCtrl,
                  decoration: const InputDecoration(labelText: 'Descuento')),
              TextField(
                  controller: totalCtrl,
                  decoration: const InputDecoration(labelText: 'Total')),
              TextField(
                  controller: cashReceivedCtrl,
                  decoration: const InputDecoration(labelText: 'Recibido')),
              TextField(
                  controller: changeCtrl,
                  decoration: const InputDecoration(labelText: 'Vuelto')),
            ],
          ),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancelar')),
          ElevatedButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Guardar')),
        ],
      ),
    );

    if (ok != true) return;

    final auditReason = await _requestAuditReason('Motivo de edicion');
    if (auditReason == null) return;

    try {
      await _service.updateSale(
        branchId: branchId,
        saleId: sale.saleId,
        metodoPago: metodoCtrl.text.trim(),
        subtotal: double.tryParse(subtotalCtrl.text.trim()) ?? sale.subtotal,
        discount: double.tryParse(discountCtrl.text.trim()) ?? sale.discount,
        total: double.tryParse(totalCtrl.text.trim()) ?? sale.total,
        cashReceived:
            double.tryParse(cashReceivedCtrl.text.trim()) ?? sale.cashReceived,
        changeAmount:
            double.tryParse(changeCtrl.text.trim()) ?? sale.changeAmount,
        customerId: sale.customerId,
        auditReason: auditReason,
      );

      if (!mounted) return;
      _showInfo('Venta actualizada correctamente.');
      await _load();
    } catch (e) {
      if (!mounted) return;
      _showInfo(_cleanError(e));
    }
  }

  Future<void> _handleDeleteSale(CashSaleModel sale) async {
    final branchId = _selectedBranchId;
    if (branchId == null) return;

    final ok = await SuperGodsUi.confirm(
      context,
      title: 'Eliminar venta',
      message:
          'Esta acción retirará la venta y quedará registrada en auditoría. No se puede deshacer.',
      confirmLabel: 'Eliminar',
      icon: Icons.delete_outline_rounded,
      destructive: true,
    );

    if (ok != true) return;

    final auditReason = await _requestAuditReason('Motivo de eliminacion');
    if (auditReason == null) return;

    try {
      await _service.deleteSale(
        branchId: branchId,
        saleId: sale.saleId,
        auditReason: auditReason,
      );

      if (!mounted) return;
      _showInfo('Venta eliminada correctamente.');
      await _load();
    } catch (e) {
      if (!mounted) return;
      _showInfo(_cleanError(e));
    }
  }

  Future<void> _showReconciliationDialog() async {
    final pending = _pendingReconciliation;
    final branchId = _selectedBranchId;
    if (pending == null || branchId == null) return;

    String paymentMethodLabel(String code) {
      final normalized = normalizePaymentCode(code);
      for (final method in _paymentMethods) {
        if (normalizePaymentCode(method.code) == normalized &&
            method.displayName.trim().isNotEmpty) {
          return method.displayName.trim();
        }
      }
      switch (normalized) {
        case 'CASH':
          return 'Efectivo';
        case 'CARD':
          return 'Tarjeta';
        case 'YAPE':
          return 'Yape';
        case 'PLIN':
          return 'Plin';
        default:
          return normalized.replaceAll('_', ' ');
      }
    }

    final saved = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      backgroundColor: Colors.transparent,
      barrierColor: const Color(0xB8000000),
      builder: (_) => CashReconciliationSheet(
        pending: pending,
        canManageFund: _canManageFund,
        paymentMethodLabel: paymentMethodLabel,
        cleanError: _cleanError,
        onSubmit: ({
          required countedAmount,
          required fundDeposits,
          note,
        }) async {
          await _service.reconcileCashRegister(
            branchId: branchId,
            cashRegisterId: pending.id,
            closingAmountCounted: countedAmount,
            fundDeposits: fundDeposits,
            note: note,
          );
        },
      ),
    );

    if (saved == true && mounted) await _load();
  }

  @override
  Widget build(BuildContext context) {
    const gold = Color(0xFFD4AF37);
    const bg = Color(0xFFF6F3ED);

    return Scaffold(
      backgroundColor: bg,
      appBar: AppBar(
        title: Text(context.ownerCoreCopy.text('cash')),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black87,
        elevation: 0.6,
        actions: [
          IconButton(
            tooltip: context.ownerCoreCopy.text('refresh'),
            onPressed: _loading || _navigating ? null : _load,
            icon: const Icon(Icons.sync_rounded),
          ),
          IconButton(
            tooltip: context.ownerCoreCopy.text('history'),
            onPressed: _goCashHistory,
            icon: const Icon(Icons.history_rounded),
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _initialize,
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : _error != null
                ? ListView(
                    physics: const AlwaysScrollableScrollPhysics(),
                    padding: const EdgeInsets.all(16),
                    children: [
                      if (_branches.isNotEmpty) ...[
                        _BranchSelectorCard(
                          branches: _branches,
                          selectedBranchId:
                              _selectedBranchId ?? _branches.first.id,
                          onChanged: _navigating ? (_) {} : _changeBranch,
                        ),
                        const SizedBox(height: 12),
                      ],
                      _ErrorCard(message: _error!, onRetry: _initialize),
                    ],
                  )
                : _branches.isEmpty
                    ? ListView(
                        physics: const AlwaysScrollableScrollPhysics(),
                        padding: const EdgeInsets.all(16),
                        children: [
                          _ErrorCard(
                            message:
                                'No tienes sedes registradas para este negocio.',
                            onRetry: _initialize,
                          ),
                        ],
                      )
                    : ListView(
                        physics: const AlwaysScrollableScrollPhysics(),
                        padding: const EdgeInsets.all(16),
                        children: [
                          if (_pendingValidationSales.isNotEmpty) ...[
                            _PendingPaymentsCompactButton(
                              count: _pendingValidationSales.length,
                              onTap: _goPendingPayments,
                            ),
                            const SizedBox(height: 12),
                          ],
                          _BranchSelectorCard(
                            branches: _branches,
                            selectedBranchId:
                                _selectedBranchId ?? _branches.first.id,
                            onChanged: _navigating ? (_) {} : _changeBranch,
                          ),
                          const SizedBox(height: 10),
                          _LiveSyncChip(
                            lastUpdatedAt: _lastUpdatedAt,
                            refreshing: _silentRefreshing,
                          ),
                          const SizedBox(height: 14),
                          if (_pendingReconciliation != null) ...[
                            _PendingReconciliationCard(
                              pending: _pendingReconciliation!,
                              canManageFund: _canManageFund,
                              onPressed: _showReconciliationDialog,
                            ),
                            const SizedBox(height: 14),
                          ],
                          _CashHeaderCard(
                              cash: _cash,
                              gold: gold,
                              branchName: _selectedBranchName),
                          const SizedBox(height: 14),
                          _ActionGrid(
                            cashOpened: _cash != null,
                            busy: _navigating,
                            onOpen: _goOpenCash,
                            onClose: _goCloseCash,
                            onSale: _goNewSale,
                            onMovement: () => _showMovementSheet(),
                            onFund: _showFundSheet,
                            canManageFund: _canManageFund,
                            onBarberPayment: _goBarberPayment,
                          ),
                          const SizedBox(height: 10),
                          _LocalConsumptionOrdersSection(
                            orders: _localConsumptionOrders,
                            cashOpened: _cash != null,
                            processingOrderId:
                                _processingLocalConsumptionOrderId,
                            professionalLabel:
                                _ownerLabels.professionalSingular,
                            onReject: (order) =>
                                _handleLocalConsumptionOrderAction(
                              order,
                              'reject',
                            ),
                            onComplete: (order) =>
                                _handleLocalConsumptionOrderAction(
                              order,
                              'complete',
                            ),
                          ),
                          if (_localConsumptionOrders.isNotEmpty)
                            const SizedBox(height: 18),
                          _ProductOrdersSection(
                            orders: _productOrders,
                            cashOpened: _cash != null,
                            processingOrderId: _processingProductOrderId,
                            onApprove: (order) =>
                                _handleProductOrderAction(order, 'approve'),
                            onReject: (order) =>
                                _handleProductOrderAction(order, 'reject'),
                            onDeliver: (order) =>
                                _handleProductOrderAction(order, 'deliver'),
                          ),
                          const SizedBox(height: 18),
                          if (_cash != null) ...[
                            _SummarySection(
                              cash: _cash!,
                              gold: gold,
                              canManageFund: _canManageFund,
                            ),
                            const SizedBox(height: 12),
                            _CourtesySummaryCard(
                              count: _courtesyCutsCount,
                              referenceValue: _courtesyReferenceValue,
                              byBarber: _courtesyByBarber,
                              professionalLabel:
                                  _ownerLabels.professionalSingular,
                            ),
                            const SizedBox(height: 12),
                            _PaymentMethodBalancesCard(
                              balances: _cash!.paymentMethodBalances.isNotEmpty
                                  ? _cash!.paymentMethodBalances
                                  : _cash!.paymentMethodsSummary,
                              paymentMethods: _paymentMethods,
                            ),
                            const SizedBox(height: 18),
                            _MovementSection(
                              cash: _cash!,
                              canManageFund: _canManageFund,
                              onAdd: () => _showMovementSheet(),
                              onEdit: (movement) =>
                                  _showMovementSheet(movement: movement),
                              onDelete: _handleDeleteMovement,
                            ),
                            const SizedBox(height: 18),
                            _SalesSection(
                              sales: _sales,
                              canManageSales: _canManageSales,
                              onEditSale: _handleEditSale,
                              onDeleteSale: _handleDeleteSale,
                            ),
                          ] else
                            _EmptyCashSection(
                                onOpen: _goOpenCash,
                                branchName: _selectedBranchName),
                        ],
                      ),
      ),
    );
  }

  Future<void> _goPendingPayments() async {
    final branchId = _selectedBranchId;
    if (branchId == null) return;

    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerPendingPaymentsScreen(
          branchId: branchId,
          branchName: _selectedBranchName,
        ),
      ),
    );

    if (!mounted) return;
    await _load();
  }

  Future<void> _goCashHistory() async {
    final branchId = _selectedBranchId;
    if (branchId == null) return;

    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerCashHistoryScreen(
          branchId: branchId,
          branchName: _selectedBranchName,
        ),
      ),
    );

    if (!mounted) return;
    await _load();
  }

  Future<void> _goBarberPayment() async {
    final branchId = _selectedBranchId;
    if (branchId == null) return;

    await _runNavigation(() async {
      await _load();

      if (!mounted) return;

      if (_cash == null) {
        _showInfo('Primero abre una caja en esta sede.');
        return;
      }

      await Navigator.push<bool>(
        context,
        MaterialPageRoute(
          builder: (_) => OwnerBarberPaymentScreen(
            branchId: branchId,
            branchName: _selectedBranchName,
            cashRegisterId: _cash!.id,
            canManageFund: _canManageFund,
          ),
        ),
      );

      if (!mounted) return;
      await _load();
    });
  }
}

class _PendingPaymentsCompactButton extends StatelessWidget {
  final int count;
  final VoidCallback onTap;

  const _PendingPaymentsCompactButton(
      {required this.count, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: const Color(0xFFFFFBEB),
      borderRadius: BorderRadius.circular(18),
      child: InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 13),
          decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: const Color(0xFFFDE68A))),
          child: Row(children: [
            const CircleAvatar(
                radius: 18,
                backgroundColor: Color(0xFFFEF3C7),
                child: Icon(Icons.verified_rounded,
                    color: Color(0xFF92400E), size: 19)),
            const SizedBox(width: 12),
            Expanded(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Text(
                      count == 1
                          ? '1 servicio por aprobar'
                          : '$count servicios por aprobar',
                      style: const TextStyle(
                          color: Color(0xFF78350F),
                          fontSize: 14.5,
                          fontWeight: FontWeight.w900)),
                  const SizedBox(height: 2),
                  const Text('Revisa el pago antes de sumarlo a caja',
                      style: TextStyle(
                          color: Color(0xFF92400E),
                          fontSize: 11.5,
                          fontWeight: FontWeight.w700)),
                ])),
            Container(
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
                decoration: BoxDecoration(
                    color: const Color(0xFFDC2626),
                    borderRadius: BorderRadius.circular(999)),
                child: Text(count > 99 ? '99+' : '$count',
                    style: const TextStyle(
                        color: Colors.white, fontWeight: FontWeight.w900))),
            const SizedBox(width: 8),
            Text(context.ownerCoreCopy.text('view'),
                style: const TextStyle(
                    color: Color(0xFF92400E), fontWeight: FontWeight.w900)),
            const SizedBox(width: 4),
            const Icon(Icons.chevron_right_rounded, color: Color(0xFF92400E)),
          ]),
        ),
      ),
    );
  }
}

class _CourtesyEntry {
  final String barberName;
  final int count;
  final double referenceValue;

  const _CourtesyEntry({
    required this.barberName,
    required this.count,
    required this.referenceValue,
  });
}

class _BarberCourtesySummary {
  final String barberName;
  final int count;
  final double referenceValue;

  const _BarberCourtesySummary({
    required this.barberName,
    required this.count,
    required this.referenceValue,
  });

  _BarberCourtesySummary copyWith({
    String? barberName,
    int? count,
    double? referenceValue,
  }) {
    return _BarberCourtesySummary(
      barberName: barberName ?? this.barberName,
      count: count ?? this.count,
      referenceValue: referenceValue ?? this.referenceValue,
    );
  }
}

class _CourtesySummaryCard extends StatelessWidget {
  final int count;
  final double referenceValue;
  final List<_BarberCourtesySummary> byBarber;
  final String professionalLabel;

  const _CourtesySummaryCard({
    required this.count,
    required this.referenceValue,
    required this.byBarber,
    required this.professionalLabel,
  });

  @override
  Widget build(BuildContext context) {
    final hasCourtesy = count > 0;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: hasCourtesy ? const Color(0xFFFFFBEB) : Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color:
              hasCourtesy ? const Color(0xFFF4D58D) : const Color(0xFFE8DEC7),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.035),
            blurRadius: 12,
            offset: const Offset(0, 5),
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
                  color: hasCourtesy
                      ? const Color(0xFFFFF3C4)
                      : const Color(0xFFF8F4EA),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(
                  Icons.redeem_rounded,
                  color: hasCourtesy
                      ? const Color(0xFF8A6A00)
                      : const Color(0xFF8B7A4E),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      context.ownerCoreCopy.text('courtesiesToday'),
                      style: const TextStyle(
                        color: Color(0xFF17131E),
                        fontSize: 16,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      context.ownerCoreCopy.text('courtesyInternalOnly'),
                      style: const TextStyle(
                        color: Color(0xFF7A7282),
                        fontSize: 12.5,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: _CourtesyMetricBox(
                  label: context.ownerCoreCopy.text('freeServices'),
                  value: '$count',
                  icon: Icons.content_cut_rounded,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _CourtesyMetricBox(
                  label: context.ownerCoreCopy.text('referenceValue'),
                  value:
                      '${TenantMoneyService.format(referenceValue, decimals: 2)}',
                  icon: Icons.price_check_rounded,
                ),
              ),
            ],
          ),
          if (byBarber.isNotEmpty) ...[
            const SizedBox(height: 14),
            Text(
              'Resumen por $professionalLabel',
              style: const TextStyle(
                color: Color(0xFF17131E),
                fontSize: 13.5,
                fontWeight: FontWeight.w900,
              ),
            ),
            const SizedBox(height: 8),
            ...byBarber.map(
              (item) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: const Color(0xFFE9E1D0)),
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          item.barberName,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            color: Color(0xFF17131E),
                            fontWeight: FontWeight.w900,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        '${item.count} cortesía${item.count == 1 ? '' : 's'}',
                        style: const TextStyle(
                          color: Color(0xFF8A6A00),
                          fontWeight: FontWeight.w900,
                          fontSize: 12.5,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        '${TenantMoneyService.format(item.referenceValue, decimals: 2)}',
                        style: const TextStyle(
                          color: Color(0xFF7A7282),
                          fontWeight: FontWeight.w800,
                          fontSize: 12.5,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _CourtesyMetricBox extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;

  const _CourtesyMetricBox({
    required this.label,
    required this.value,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE9E1D0)),
      ),
      child: Row(
        children: [
          Icon(icon, color: const Color(0xFF8A6A00), size: 19),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    color: Color(0xFF7A7282),
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  value,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    color: Color(0xFF17131E),
                    fontSize: 15,
                    fontWeight: FontWeight.w900,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _LiveSyncChip extends StatelessWidget {
  final DateTime? lastUpdatedAt;
  final bool refreshing;

  const _LiveSyncChip({
    required this.lastUpdatedAt,
    required this.refreshing,
  });

  String _timeText(BuildContext context) {
    final date = lastUpdatedAt;
    if (date == null) return context.ownerCoreCopy.text('syncingData');

    final hh = date.hour.toString().padLeft(2, '0');
    final mm = date.minute.toString().padLeft(2, '0');
    return context.ownerCoreCopy.withValue('updatedAt', '$hh:$mm');
  }

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 7),
          decoration: BoxDecoration(
            color: const Color(0xFFEFFDF4),
            borderRadius: BorderRadius.circular(999),
            border: Border.all(color: const Color(0xFFBBF7D0)),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 7,
                height: 7,
                decoration: const BoxDecoration(
                  color: Color(0xFF16A34A),
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 7),
              Text(
                context.ownerCoreCopy.withValue(
                  'liveUpdated',
                  refreshing
                      ? context.ownerCoreCopy.text('updating')
                      : _timeText(context),
                ),
                style: const TextStyle(
                  color: Color(0xFF15803D),
                  fontWeight: FontWeight.w900,
                  fontSize: 11.5,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _BranchSelectorCard extends StatelessWidget {
  final List<BranchOptionModel> branches;
  final int selectedBranchId;
  final ValueChanged<int> onChanged;

  const _BranchSelectorCard({
    required this.branches,
    required this.selectedBranchId,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    final selected = branches.firstWhere(
      (b) => b.id == selectedBranchId,
      orElse: () => BranchOptionModel(id: 0, name: 'Sede'),
    );

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFEBDFAF)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            context.ownerCoreCopy.text('activeBranch'),
            style: const TextStyle(
              fontSize: 11.5,
              fontWeight: FontWeight.w700,
              color: Color(0xFF9C8E6A),
            ),
          ),
          const SizedBox(height: 6),
          Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: const Color(0xFFF8F4EA),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.storefront_outlined,
                    color: Color(0xFF8B7A4E), size: 20),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  selected.name.trim().isEmpty
                      ? 'Sede ${selected.id}'
                      : selected.name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w900,
                    color: Color(0xFF1A1A1A),
                    letterSpacing: 0.2,
                  ),
                ),
              ),
              PopupMenuButton<int>(
                tooltip: context.ownerCoreCopy.text('change'),
                onSelected: onChanged,
                itemBuilder: (_) => branches
                    .map(
                      (b) => PopupMenuItem<int>(
                        value: b.id,
                        child: Row(
                          children: [
                            Icon(
                              b.id == selectedBranchId
                                  ? Icons.check_circle_rounded
                                  : Icons.store_mall_directory_outlined,
                              size: 18,
                              color: b.id == selectedBranchId
                                  ? const Color(0xFFD4AF37)
                                  : const Color(0xFF667085),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                b.name.trim().isEmpty ? 'Sede ${b.id}' : b.name,
                                style: TextStyle(
                                  fontWeight: b.id == selectedBranchId
                                      ? FontWeight.w800
                                      : FontWeight.w600,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    )
                    .toList(),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14)),
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF8F4EA),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: const Color(0xFFEBDFAF)),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        context.ownerCoreCopy.text('change'),
                        style: const TextStyle(
                            fontSize: 12.5,
                            fontWeight: FontWeight.w800,
                            color: Color(0xFF6B5B2A)),
                      ),
                      const SizedBox(width: 6),
                      const Icon(Icons.keyboard_arrow_down_rounded,
                          size: 18, color: Color(0xFF6B5B2A)),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _PendingReconciliationCard extends StatelessWidget {
  final CashRegisterModel pending;
  final bool canManageFund;
  final VoidCallback onPressed;

  const _PendingReconciliationCard({
    required this.pending,
    required this.canManageFund,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    const amber = Color(0xFF8A6100);
    const navy = Color(0xFF182033);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFFFFBEB),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFFE7C75B)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x147A5700),
            blurRadius: 16,
            offset: Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: const Color(0xFFFFE8A3),
                  borderRadius: BorderRadius.circular(13),
                ),
                child: const Icon(
                  Icons.priority_high_rounded,
                  color: amber,
                  size: 24,
                ),
              ),
              const SizedBox(width: 11),
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'ACCIÓN NECESARIA',
                      style: TextStyle(
                        color: amber,
                        fontSize: 10,
                        fontWeight: FontWeight.w900,
                        letterSpacing: 1.1,
                      ),
                    ),
                    SizedBox(height: 3),
                    Text(
                      'Revisa el cierre anterior',
                      style: TextStyle(
                        color: navy,
                        fontSize: 17,
                        height: 1.15,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 13),
          const Text(
            'La caja anterior se cerró automáticamente. Cuenta el dinero '
            'que encontraste y compáralo con este saldo:',
            style: TextStyle(
              color: Color(0xFF4B5563),
              fontSize: 12.5,
              height: 1.4,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 11),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 11),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: const Color(0xFFF0D77F)),
            ),
            child: Row(
              children: [
                const Icon(
                  Icons.account_balance_wallet_rounded,
                  color: amber,
                  size: 21,
                ),
                const SizedBox(width: 9),
                const Expanded(
                  child: Text(
                    'Saldo esperado',
                    style: TextStyle(
                      color: Color(0xFF6B7280),
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
                Text(
                  TenantMoneyService.format(
                    pending.closingAmountExpected,
                    decimals: 2,
                  ),
                  style: const TextStyle(
                    color: navy,
                    fontSize: 18,
                    fontWeight: FontWeight.w900,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 9),
          Text(
            canManageFund
                ? 'Al revisar podrás registrar diferencias y decidir cuánto enviar al fondo.'
                : 'Al revisar podrás confirmar el monto y registrar cualquier diferencia.',
            style: const TextStyle(
              color: Color(0xFF6B7280),
              fontSize: 11,
              height: 1.35,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 13),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: onPressed,
              style: FilledButton.styleFrom(
                backgroundColor: navy,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 13),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(14),
                ),
              ),
              icon: const Icon(Icons.fact_check_rounded, size: 20),
              label: const Text(
                'Revisar y conciliar cierre',
                style: TextStyle(fontWeight: FontWeight.w900),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _CashHeaderCard extends StatelessWidget {
  final CashRegisterModel? cash;
  final Color gold;
  final String branchName;

  const _CashHeaderCard({
    required this.cash,
    required this.gold,
    required this.branchName,
  });

  @override
  Widget build(BuildContext context) {
    final isOpen = cash != null;

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        gradient: LinearGradient(
          colors: isOpen
              ? [const Color(0xFF1F1A10), const Color(0xFF3A2F14)]
              : [const Color(0xFF3B3B3B), const Color(0xFF5A5A5A)],
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.10),
            blurRadius: 18,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 26,
            backgroundColor: gold,
            child: Icon(
                isOpen ? Icons.point_of_sale_rounded : Icons.lock_clock_rounded,
                color: Colors.black87),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  isOpen
                      ? context.ownerCoreCopy.withValue('cashOpen', branchName)
                      : context.ownerCoreCopy
                          .withValue('cashClosed', branchName),
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 19,
                      fontWeight: FontWeight.w800),
                ),
                const SizedBox(height: 4),
                Text(
                  isOpen
                      ? context.ownerCoreCopy.withValue(
                          'responsible',
                          cash!.assignedUserName ??
                              cash!.openedByUserName ??
                              context.ownerCoreCopy.text('unassigned'),
                        )
                      : context.ownerCoreCopy.text('startCharging'),
                  style: const TextStyle(color: Colors.white70, fontSize: 13.5),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ActionGrid extends StatelessWidget {
  final bool cashOpened;
  final bool busy;
  final VoidCallback onOpen;
  final VoidCallback onClose;
  final VoidCallback onSale;
  final VoidCallback onMovement;
  final VoidCallback onFund;
  final bool canManageFund;
  final VoidCallback onBarberPayment;

  const _ActionGrid({
    required this.cashOpened,
    required this.busy,
    required this.onOpen,
    required this.onClose,
    required this.onSale,
    required this.onMovement,
    required this.onFund,
    required this.canManageFund,
    required this.onBarberPayment,
  });

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 10,
      runSpacing: 10,
      children: [
        SizedBox(
            width: (MediaQuery.of(context).size.width - 42) / 2,
            child: _ActionButton(
                icon: Icons.lock_open_rounded,
                label: context.ownerCoreCopy.text('openCash'),
                enabled: !cashOpened && !busy,
                onTap: onOpen)),
        SizedBox(
            width: (MediaQuery.of(context).size.width - 42) / 2,
            child: _ActionButton(
                icon: Icons.add_shopping_cart_rounded,
                label: context.ownerCoreCopy.text('newSale'),
                enabled: cashOpened && !busy,
                onTap: onSale)),
        SizedBox(
            width: (MediaQuery.of(context).size.width - 42) / 2,
            child: _ActionButton(
                icon: Icons.receipt_long_rounded,
                label: context.ownerCoreCopy.text('moveCash'),
                enabled: cashOpened && !busy,
                onTap: onMovement)),
        if (canManageFund)
          SizedBox(
              width: (MediaQuery.of(context).size.width - 42) / 2,
              child: _ActionButton(
                  icon: Icons.savings_outlined,
                  label: context.ownerCoreCopy.text('manageFund'),
                  enabled: !busy,
                  onTap: onFund)),
        SizedBox(
            width: (MediaQuery.of(context).size.width - 42) / 2,
            child: _ActionButton(
                icon: Icons.task_alt_rounded,
                label: context.ownerCoreCopy.text('closeCash'),
                enabled: cashOpened && !busy,
                onTap: onClose)),
        SizedBox(
            width: (MediaQuery.of(context).size.width - 42) / 2,
            child: _ActionButton(
                icon: Icons.payments_rounded,
                label: context.ownerCoreCopy.text('payProfessional'),
                enabled: cashOpened && !busy,
                onTap: onBarberPayment)),
      ],
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool enabled;
  final VoidCallback onTap;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.enabled,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    const gold = Color(0xFFD4AF37);
    const border = Color(0xFFE8DEC7);

    return GestureDetector(
      onTap: enabled ? onTap : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 220),
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 8),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(20),
          gradient: enabled
              ? const LinearGradient(
                  colors: [Color(0xFF17131E), Color(0xFF2A2137)])
              : null,
          color: enabled ? null : Colors.white,
          border: Border.all(color: enabled ? Colors.transparent : border),
          boxShadow: enabled
              ? [
                  BoxShadow(
                      color: Colors.black.withOpacity(0.10),
                      blurRadius: 16,
                      offset: const Offset(0, 7)),
                ]
              : [],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon,
                color: enabled ? gold : const Color(0xFFB7B0BC), size: 22),
            const SizedBox(height: 8),
            Text(
              label,
              textAlign: TextAlign.center,
              style: TextStyle(
                  fontSize: 12.5,
                  fontWeight: FontWeight.w700,
                  color: enabled ? Colors.white : const Color(0xFFB7B0BC)),
            ),
          ],
        ),
      ),
    );
  }
}

class _SummarySection extends StatelessWidget {
  final CashRegisterModel cash;
  final Color gold;
  final bool canManageFund;

  const _SummarySection({
    required this.cash,
    required this.gold,
    required this.canManageFund,
  });

  @override
  Widget build(BuildContext context) {
    final expected = cash.closingAmountExpected;
    final isNegative = expected < 0;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (isNegative) ...[
          _NegativeCashAlert(
            cashSalesTotal: cash.cashSalesTotal,
            movementsExpense: cash.movementsExpense,
            expected: cash.closingAmountExpected,
          ),
          const SizedBox(height: 10),
        ],
        _CashQuickOverviewCard(
          expected: cash.closingAmountExpected,
          openingAmount: cash.openingAmount,
          cashSalesTotal: cash.cashSalesTotal,
          movementsIncome: cash.movementsIncome,
          cashMovementsExpense: cash.cashMovementsExpense,
          isNegative: isNegative,
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            if (canManageFund) ...[
              Expanded(
                child: _CompactMetricCard(
                  title: context.ownerCoreCopy.text('accumulatedFund'),
                  value: TenantMoneyService.format(
                    cash.accumulatedFundBalance,
                    decimals: 2,
                  ),
                  icon: Icons.savings_rounded,
                  valueColor: const Color(0xFF8A6B00),
                ),
              ),
              const SizedBox(width: 8),
            ],
            Expanded(
              child: _CompactMetricCard(
                title: context.ownerCoreCopy.text('todayCash'),
                value: TenantMoneyService.format(
                  cash.closingAmountExpected,
                  decimals: 2,
                ),
                icon: Icons.account_balance_wallet_rounded,
                valueColor: cash.closingAmountExpected < 0
                    ? const Color(0xFFB91C1C)
                    : const Color(0xFF15803D),
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: _CompactMetricCard(
                title: context.ownerCoreCopy.text('cashSales'),
                value:
                    '${TenantMoneyService.format(cash.cashSalesTotal, decimals: 2)}',
                icon: Icons.payments_rounded,
                valueColor: const Color(0xFF15803D),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _CompactMetricCard(
                title: context.ownerCoreCopy.text('income'),
                value:
                    '${TenantMoneyService.format(cash.movementsIncome, decimals: 2)}',
                icon: Icons.add_circle_outline_rounded,
                valueColor: const Color(0xFF15803D),
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: _CompactMetricCard(
                title: context.ownerCoreCopy.text('outflowsTotal'),
                value:
                    '${TenantMoneyService.format(cash.movementsExpense, decimals: 2)}',
                icon: Icons.remove_circle_outline_rounded,
                valueColor: const Color(0xFFC2410C),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _CompactMetricCard(
                title: context.ownerCoreCopy.text('opening'),
                value:
                    '${TenantMoneyService.format(cash.openingAmount, decimals: 2)}',
                icon: Icons.lock_open_rounded,
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        _CashDetailsExpansion(cash: cash),
      ],
    );
  }
}

class _CashQuickOverviewCard extends StatelessWidget {
  final double expected;
  final double openingAmount;
  final double cashSalesTotal;
  final double movementsIncome;
  final double cashMovementsExpense;
  final bool isNegative;

  const _CashQuickOverviewCard({
    required this.expected,
    required this.openingAmount,
    required this.cashSalesTotal,
    required this.movementsIncome,
    required this.cashMovementsExpense,
    required this.isNegative,
  });

  @override
  Widget build(BuildContext context) {
    final expectedColor =
        isNegative ? const Color(0xFFFCA5A5) : const Color(0xFF86EFAC);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(22),
        gradient: const LinearGradient(
          colors: [
            Color(0xFF0F172A),
            Color(0xFF17131E),
            Color(0xFF241B2F),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.10),
            blurRadius: 18,
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
                width: 38,
                height: 38,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.10),
                  borderRadius: BorderRadius.circular(13),
                ),
                child: Icon(
                  isNegative
                      ? Icons.warning_amber_rounded
                      : Icons.account_balance_wallet_rounded,
                  color: isNegative
                      ? const Color(0xFFFCA5A5)
                      : const Color(0xFFD4AF37),
                  size: 21,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      context.ownerCoreCopy.text('quickSummary'),
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 15.5,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      context.ownerCoreCopy.text('physicalCashExpectedNow'),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        color: Color(0xFFCBD5E1),
                        fontSize: 11.5,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
              Text(
                '${TenantMoneyService.format(expected, decimals: 2)}',
                style: TextStyle(
                  color: expectedColor,
                  fontSize: 21,
                  fontWeight: FontWeight.w900,
                  letterSpacing: -0.3,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.08),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.white.withOpacity(0.08)),
            ),
            child: Text(
              '${context.ownerCoreCopy.text('opening')} ${TenantMoneyService.format(openingAmount, decimals: 2)} + ${context.ownerCoreCopy.text('cashSales').toLowerCase()} ${TenantMoneyService.format(cashSalesTotal, decimals: 2)} + ${context.ownerCoreCopy.text('income').toLowerCase()} ${TenantMoneyService.format(movementsIncome, decimals: 2)} - ${context.ownerCoreCopy.text('cashOutflows').toLowerCase()} ${TenantMoneyService.format(cashMovementsExpense, decimals: 2)}',
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: Color(0xFFE5E7EB),
                fontSize: 11.7,
                fontWeight: FontWeight.w800,
                height: 1.25,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _CompactMetricCard extends StatelessWidget {
  final String title;
  final String value;
  final IconData icon;
  final Color? valueColor;

  const _CompactMetricCard({
    required this.title,
    required this.value,
    required this.icon,
    this.valueColor,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFF0E7D3)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.035),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 32,
            height: 32,
            decoration: BoxDecoration(
              color: const Color(0xFFF8F4EA),
              borderRadius: BorderRadius.circular(11),
            ),
            child: Icon(
              icon,
              color: const Color(0xFF8B7A4E),
              size: 18,
            ),
          ),
          const SizedBox(width: 9),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 11.2,
                    fontWeight: FontWeight.w800,
                    color: Color(0xFF7A7282),
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  value,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: 13.8,
                    fontWeight: FontWeight.w900,
                    color: valueColor ?? const Color(0xFF17131E),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _CashDetailsExpansion extends StatelessWidget {
  final CashRegisterModel cash;

  const _CashDetailsExpansion({required this.cash});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFF0E7D3)),
      ),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          tilePadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 0),
          childrenPadding: const EdgeInsets.fromLTRB(14, 0, 14, 14),
          iconColor: const Color(0xFF17131E),
          collapsedIconColor: const Color(0xFF7A7282),
          title: Text(
            context.ownerCoreCopy.text('outflowBreakdown'),
            style: const TextStyle(
              fontSize: 12.8,
              fontWeight: FontWeight.w900,
              color: Color(0xFF17131E),
            ),
          ),
          subtitle: Text(
            context.ownerCoreCopy.withValue('expensesAdvancesPayments',
                _ownerLabels.professionalsPluralLower),
            style: TextStyle(
              fontSize: 11.3,
              fontWeight: FontWeight.w700,
              color: Color(0xFF7A7282),
            ),
          ),
          children: [
            Row(
              children: [
                Expanded(
                  child: _CompactBreakdownPill(
                    label: context.ownerCoreCopy.text('cashOutflows'),
                    value: cash.cashMovementsExpense,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _CompactBreakdownPill(
                    label: context.ownerCoreCopy.text('otherMethodOutflows'),
                    value: (cash.movementsExpense - cash.cashMovementsExpense)
                        .clamp(0, double.infinity)
                        .toDouble(),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: _CompactBreakdownPill(
                    label: context.ownerCoreCopy.text('expenses'),
                    value: cash.movementsExpenseGeneral,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _CompactBreakdownPill(
                    label: context.ownerCoreCopy.text('advances'),
                    value: cash.movementsAdvanceBarber,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: _CompactBreakdownPill(
                    label: context.ownerCoreCopy.withValue(
                        'professionalPayment',
                        _ownerLabels.professionalSingularLower),
                    value: cash.movementsPaymentBarber,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _CompactBreakdownPill(
                    label: context.ownerCoreCopy.text('expectedCash'),
                    value: cash.closingAmountExpected,
                    isExpected: true,
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

class _CompactBreakdownPill extends StatelessWidget {
  final String label;
  final double value;
  final bool isExpected;

  const _CompactBreakdownPill({
    required this.label,
    required this.value,
    this.isExpected = false,
  });

  @override
  Widget build(BuildContext context) {
    final color = isExpected
        ? (value < 0 ? const Color(0xFFB91C1C) : const Color(0xFF17131E))
        : const Color(0xFFC2410C);

    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: isExpected ? const Color(0xFFFFFBEB) : const Color(0xFFF8FAFC),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: isExpected ? const Color(0xFFFDE68A) : const Color(0xFFE5E7EB),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              color: Color(0xFF7A7282),
              fontSize: 10.8,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            '${TenantMoneyService.format(value, decimals: 2)}',
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              color: color,
              fontSize: 12.8,
              fontWeight: FontWeight.w900,
            ),
          ),
        ],
      ),
    );
  }
}

class _NegativeCashAlert extends StatelessWidget {
  final double cashSalesTotal;
  final double movementsExpense;
  final double expected;

  const _NegativeCashAlert({
    required this.cashSalesTotal,
    required this.movementsExpense,
    required this.expected,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF1F2),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFFECACA)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons.warning_amber_rounded,
            color: Color(0xFFB91C1C),
            size: 24,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  context.ownerCoreCopy.text('negativeExpectedCash'),
                  style: const TextStyle(
                    color: Color(0xFF991B1B),
                    fontSize: 14.5,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                const SizedBox(height: 5),
                Text(
                  context.ownerCoreCopy.text('negativeExpectedCashDescription'),
                  style: const TextStyle(
                    color: Color(0xFFB91C1C),
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    height: 1.35,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  '${TenantMoneyService.format(cashSalesTotal, decimals: 2)} - ${TenantMoneyService.format(movementsExpense, decimals: 2)} = ${TenantMoneyService.format(expected, decimals: 2)}',
                  style: const TextStyle(
                    color: Color(0xFF7F1D1D),
                    fontSize: 12.5,
                    fontWeight: FontWeight.w900,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _PaymentMethodBalancesCard extends StatelessWidget {
  final List<PaymentMethodSummaryModel> balances;
  final List<OwnerPaymentMethodModel> paymentMethods;

  const _PaymentMethodBalancesCard({
    required this.balances,
    required this.paymentMethods,
  });

  String _normalize(String value) => normalizePaymentCode(value);

  bool _isCash(String code) {
    final value = _normalize(code);
    return value == 'CASH' || value == 'EFECTIVO';
  }

  bool _isFree(String code) {
    final value = _normalize(code);
    return value == 'FREE' || value == 'GRATIS';
  }

  String _label(String code) {
    final normalized = _normalize(code);
    for (final method in paymentMethods) {
      if (_normalize(method.code) == normalized &&
          method.displayName.trim().isNotEmpty) {
        return method.displayName.trim();
      }
    }

    switch (normalized) {
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
    switch (_normalize(code)) {
      case 'YAPE':
        return Icons.phone_iphone_rounded;
      case 'PLIN':
        return Icons.bolt_rounded;
      case 'TRANSFER':
      case 'TRANSFERENCIA':
        return Icons.account_balance_rounded;
      case 'CARD':
      case 'TARJETA':
        return Icons.credit_card_rounded;
      case 'NEQUI':
      case 'DAVIPLATA':
      case 'PAGO_MOVIL':
        return Icons.account_balance_wallet_rounded;
      case 'ZELLE':
        return Icons.public_rounded;
      case 'QR':
        return Icons.qr_code_2_rounded;
      default:
        return Icons.account_balance_wallet_rounded;
    }
  }

  Color _accent(String code) {
    switch (_normalize(code)) {
      case 'YAPE':
        return const Color(0xFF8B5CF6);
      case 'PLIN':
        return const Color(0xFF38BDF8);
      case 'TRANSFER':
      case 'TRANSFERENCIA':
        return const Color(0xFF60A5FA);
      case 'CARD':
      case 'TARJETA':
        return const Color(0xFFD4AF37);
      case 'NEQUI':
      case 'DAVIPLATA':
      case 'PAGO_MOVIL':
        return const Color(0xFF34D399);
      case 'ZELLE':
        return const Color(0xFFA78BFA);
      case 'QR':
        return const Color(0xFFFBBF24);
      default:
        return const Color(0xFFCBD5E1);
    }
  }

  String _money(double value) => TenantMoneyService.format(value);

  List<PaymentMethodSummaryModel> _visibleBalances() {
    final map = <String, PaymentMethodSummaryModel>{};

    // 1) Mostrar los métodos configurados del dueño/sede aunque estén en S/ 0.00.
    final configuredCodes = paymentMethods
        .map((method) => _normalize(method.code))
        .where((code) => code.isNotEmpty && !_isCash(code) && !_isFree(code))
        .toList();

    for (final code in configuredCodes) {
      map[code] = PaymentMethodSummaryModel(
        paymentMethod: code,
        count: 0,
        totalAmount: 0,
      );
    }

    // 2) Sobrescribir/sumar los saldos reales que vienen del backend.
    for (final item in balances) {
      final code = _normalize(item.paymentMethod);
      if (code.isEmpty || _isCash(code) || _isFree(code)) continue;

      final normalizedCode = code == 'TRANSFERENCIA'
          ? 'TRANSFER'
          : code == 'TARJETA'
              ? 'CARD'
              : code;

      final current = map[normalizedCode];

      final mergedAmount = (current?.totalAmount ?? 0) + item.totalAmount;
      final mergedCount = (current?.count ?? 0) + item.count;

      map[normalizedCode] = PaymentMethodSummaryModel(
        paymentMethod: normalizedCode,
        count: mergedCount,
        totalAmount: mergedAmount.abs() < 0.01 ? 0 : mergedAmount,
      );
    }

    final order = [
      'YAPE',
      'PLIN',
      'TRANSFER',
      'CARD',
      'NEQUI',
      'DAVIPLATA',
      'PAGO_MOVIL',
      'ZELLE',
      'QR',
    ];

    final result = map.values.toList();
    result.sort((a, b) {
      final ai = order.indexOf(_normalize(a.paymentMethod));
      final bi = order.indexOf(_normalize(b.paymentMethod));
      final av = ai == -1 ? 999 : ai;
      final bv = bi == -1 ? 999 : bi;
      if (av != bv) return av.compareTo(bv);
      return _label(a.paymentMethod).compareTo(_label(b.paymentMethod));
    });

    return result;
  }

  @override
  Widget build(BuildContext context) {
    final visible = _visibleBalances();
    final total =
        visible.fold<double>(0, (sum, item) => sum + item.totalAmount);
    final operations = visible.fold<int>(0, (sum, item) => sum + item.count);

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        gradient: const LinearGradient(
          colors: [Color(0xFF0F172A), Color(0xFF151827), Color(0xFF221A2F)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.10),
            blurRadius: 18,
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
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.10),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: const Icon(
                  Icons.account_balance_wallet_rounded,
                  color: Color(0xFFD4AF37),
                  size: 22,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Saldo actual por método',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'Después de ventas, ingresos, salidas y traslados',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        color: Color(0xFFCBD5E1),
                        fontSize: 11.5,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
              Text(
                _money(total),
                style: const TextStyle(
                  color: Color(0xFF86EFAC),
                  fontSize: 16,
                  fontWeight: FontWeight.w900,
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          if (visible.isEmpty)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.08),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.white.withOpacity(0.08)),
              ),
              child: const Text(
                'Cuando tengas saldos en métodos digitales configurados, aparecerán aquí.',
                style: TextStyle(
                  color: Color(0xFFCBD5E1),
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  height: 1.3,
                ),
              ),
            )
          else
            LayoutBuilder(
              builder: (context, constraints) {
                final itemWidth = visible.length == 1
                    ? constraints.maxWidth
                    : (constraints.maxWidth - 8) / 2;

                return Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: visible
                      .map(
                        (item) => SizedBox(
                          width: itemWidth,
                          child: _PaymentBalancePill(
                            label: _label(item.paymentMethod),
                            value: _money(item.totalAmount),
                            rawValue: item.totalAmount,
                            count: item.count,
                            icon: _icon(item.paymentMethod),
                            accent: _accent(item.paymentMethod),
                          ),
                        ),
                      )
                      .toList(),
                );
              },
            ),
        ],
      ),
    );
  }
}

class _PaymentBalancePill extends StatelessWidget {
  final String label;
  final String value;
  final double rawValue;
  final int count;
  final IconData icon;
  final Color accent;

  const _PaymentBalancePill({
    required this.label,
    required this.value,
    required this.rawValue,
    required this.count,
    required this.icon,
    required this.accent,
  });

  @override
  Widget build(BuildContext context) {
    final effectiveAccent = rawValue < 0 ? const Color(0xFFFCA5A5) : accent;

    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.08),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: effectiveAccent, size: 17),
              const SizedBox(width: 5),
              Expanded(
                child: Text(
                  label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    color: Color(0xFFE5E7EB),
                    fontSize: 11.5,
                    fontWeight: FontWeight.w900,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          FittedBox(
            fit: BoxFit.scaleDown,
            alignment: Alignment.centerLeft,
            child: Text(
              value,
              maxLines: 1,
              style: TextStyle(
                color: effectiveAccent,
                fontSize: 14.5,
                fontWeight: FontWeight.w900,
              ),
            ),
          ),
          const SizedBox(height: 3),
          Text(
            '$count venta${count == 1 ? '' : 's'}',
            style: const TextStyle(
              color: Color(0xFF94A3B8),
              fontSize: 10.5,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class _MovementSection extends StatelessWidget {
  final CashRegisterModel cash;
  final bool canManageFund;
  final VoidCallback onAdd;
  final ValueChanged<CashMovementModel> onEdit;
  final ValueChanged<CashMovementModel> onDelete;

  const _MovementSection(
      {required this.cash,
      required this.canManageFund,
      required this.onAdd,
      required this.onEdit,
      required this.onDelete});

  IconData _iconForType(String type) {
    switch (type) {
      case 'INCOME':
        return Icons.add_card_rounded;
      case 'PAYMENT_METHOD_TRANSFER':
        return Icons.swap_horiz_rounded;
      case 'ADVANCE_BARBER':
        return Icons.front_hand_rounded;
      case 'PAYMENT_BARBER':
        return Icons.content_cut_rounded;
      default:
        return Icons.receipt_long_rounded;
    }
  }

  Color _chipColor(String type) {
    switch (type) {
      case 'INCOME':
        return const Color(0xFF15803D);
      case 'PAYMENT_METHOD_TRANSFER':
        return const Color(0xFF2563EB);
      case 'ADVANCE_BARBER':
        return const Color(0xFF7C3AED);
      case 'PAYMENT_BARBER':
        return const Color(0xFF0F766E);
      default:
        return const Color(0xFFC2410C);
    }
  }

  String _labelForType(String type) {
    switch (type) {
      case 'INCOME':
        return 'Ingreso';
      case 'PAYMENT_METHOD_TRANSFER':
        return 'Traslado';
      case 'ADVANCE_BARBER':
        return 'Adelanto';
      case 'PAYMENT_BARBER':
        return 'Pago ${_ownerLabels.professionalSingularLower}';
      default:
        return 'Gasto';
    }
  }

  String _methodLabel(String? code) {
    switch ((code ?? '').trim().toUpperCase()) {
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
        final clean = (code ?? '').trim();
        if (clean.isEmpty) return 'Método';
        return clean[0].toUpperCase() + clean.substring(1).toLowerCase();
    }
  }

  @override
  Widget build(BuildContext context) {
    final movements = canManageFund
        ? cash.movements
        : cash.movements
            .where((movement) => movement.fundingSource != 'ACCUMULATED_FUND')
            .toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Expanded(
                child: Text('Gastos, ingresos y salidas de caja',
                    style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w900,
                        color: Color(0xFF17131E)))),
            TextButton.icon(
                onPressed: onAdd,
                icon: const Icon(Icons.add_rounded),
                label: const Text('Registrar')),
          ],
        ),
        const SizedBox(height: 12),
        if (movements.isEmpty)
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: const Color(0xFFF0E7D3))),
            child: const Row(
              children: [
                Icon(Icons.inventory_2_outlined, color: Color(0xFFB4ACB9)),
                SizedBox(width: 10),
                Expanded(
                    child: Text(
                        'Aún no registraste ingresos, gastos ni pagos de caja hoy.',
                        style: TextStyle(
                            color: Color(0xFF6E6676),
                            fontSize: 13.5,
                            fontWeight: FontWeight.w600))),
              ],
            ),
          )
        else
          ...movements.map((movement) {
            final color = _chipColor(movement.type);
            final barberName = movement.barberUserName?.trim();
            return Container(
              margin: const EdgeInsets.only(bottom: 12),
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: const Color(0xFFF0E9D8)),
                  boxShadow: [
                    BoxShadow(
                        color: Colors.black.withOpacity(0.045),
                        blurRadius: 12,
                        offset: const Offset(0, 5))
                  ]),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                      width: 46,
                      height: 46,
                      decoration: BoxDecoration(
                          color: color.withOpacity(0.12),
                          borderRadius: BorderRadius.circular(14)),
                      child: Icon(_iconForType(movement.type), color: color)),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Expanded(
                                child: Text(movement.concept,
                                    style: const TextStyle(
                                        fontSize: 15,
                                        fontWeight: FontWeight.w900,
                                        color: Color(0xFF17131E)))),
                            Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 10, vertical: 6),
                                decoration: BoxDecoration(
                                    color: color.withOpacity(0.10),
                                    borderRadius: BorderRadius.circular(999)),
                                child: Text(_labelForType(movement.type),
                                    style: TextStyle(
                                        color: color,
                                        fontWeight: FontWeight.w800,
                                        fontSize: 11.5))),
                          ],
                        ),
                        const SizedBox(height: 6),
                        Text(
                          movement.isPaymentTransfer
                              ? '${_methodLabel(movement.fromPaymentMethod)} → ${_methodLabel(movement.toPaymentMethod)} · ${TenantMoneyService.format(movement.amount, decimals: 2)}'
                              : '${TenantMoneyService.format(movement.amount, decimals: 2)} • ${_methodLabel(movement.paymentMethod ?? 'CASH')}',
                          style: const TextStyle(
                            color: Color(0xFF7A7282),
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        if (movement.fundingSource == 'ACCUMULATED_FUND') ...[
                          const SizedBox(height: 4),
                          const Text(
                              'Origen: Fondo acumulado · no afecta el saldo de esta caja',
                              style: TextStyle(
                                  color: Color(0xFF8A6A00),
                                  fontWeight: FontWeight.w800,
                                  fontSize: 12)),
                        ] else if (movement.fundingSource == 'EXTERNAL') ...[
                          const SizedBox(height: 4),
                          const Text(
                              'Origen: Cuenta externa · no afecta esta caja',
                              style: TextStyle(
                                  color: Color(0xFF667085),
                                  fontWeight: FontWeight.w800,
                                  fontSize: 12)),
                        ],
                        if (barberName != null && barberName.isNotEmpty) ...[
                          const SizedBox(height: 4),
                          Text(
                              '${_ownerLabels.professionalSingular}: $barberName',
                              style: const TextStyle(
                                  color: Color(0xFF7A7282),
                                  fontWeight: FontWeight.w700)),
                        ],
                        if ((movement.note ?? '').trim().isNotEmpty) ...[
                          const SizedBox(height: 6),
                          Text(movement.note!.trim(),
                              style: const TextStyle(
                                  color: Color(0xFF6E6676),
                                  fontSize: 13,
                                  fontWeight: FontWeight.w500)),
                        ],
                      ],
                    ),
                  ),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      PopupMenuButton<String>(
                        padding: EdgeInsets.zero,
                        tooltip: 'Acciones',
                        icon: const Icon(Icons.more_vert_rounded,
                            size: 20, color: Color(0xFF667085)),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14)),
                        onSelected: (value) {
                          if (value == 'edit') {
                            onEdit(movement);
                          } else if (value == 'delete') {
                            onDelete(movement);
                          }
                        },
                        itemBuilder: (_) => const [
                          PopupMenuItem<String>(
                              value: 'edit',
                              child: Row(children: [
                                Icon(Icons.edit_rounded,
                                    size: 18, color: Color(0xFF17131E)),
                                SizedBox(width: 8),
                                Text('Editar')
                              ])),
                          PopupMenuItem<String>(
                              value: 'delete',
                              child: Row(children: [
                                Icon(Icons.delete_outline_rounded,
                                    size: 18, color: Color(0xFFC84C4C)),
                                SizedBox(width: 8),
                                Text('Eliminar')
                              ])),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
            );
          }),
      ],
    );
  }
}

class _LocalConsumptionOrdersSection extends StatelessWidget {
  final List<OwnerLocalConsumptionOrderModel> orders;
  final bool cashOpened;
  final int? processingOrderId;
  final String professionalLabel;
  final ValueChanged<OwnerLocalConsumptionOrderModel> onReject;
  final ValueChanged<OwnerLocalConsumptionOrderModel> onComplete;

  const _LocalConsumptionOrdersSection({
    required this.orders,
    required this.cashOpened,
    required this.processingOrderId,
    required this.professionalLabel,
    required this.onReject,
    required this.onComplete,
  });

  @override
  Widget build(BuildContext context) {
    if (orders.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: const Color(0xFFEFF6FF),
                borderRadius: BorderRadius.circular(13),
              ),
              child: const Icon(Icons.qr_code_scanner_rounded,
                  color: Color(0xFF2563EB)),
            ),
            const SizedBox(width: 10),
            const Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Consumos desde QR',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w900,
                      color: Color(0xFF17131E),
                    ),
                  ),
                  SizedBox(height: 2),
                  Text(
                    'Clientes en recepcion que eligieron servicios o productos.',
                    style: TextStyle(
                      color: Color(0xFF6B7280),
                      fontSize: 12.5,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        ...orders.map((order) {
          final busy = processingOrderId == order.id;
          final phone = order.customerPhone?.trim();

          return Container(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: const Color(0xFFBFDBFE)),
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
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: 52,
                      height: 52,
                      decoration: BoxDecoration(
                        color: const Color(0xFFEEF2FF),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: const Icon(Icons.person_pin_circle_outlined,
                          color: Color(0xFF2563EB)),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            order.customerName,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w900,
                              color: Color(0xFF111827),
                            ),
                          ),
                          if (phone != null && phone.isNotEmpty) ...[
                            const SizedBox(height: 3),
                            Text(
                              phone,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                color: Color(0xFF667085),
                                fontSize: 12.5,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ],
                          const SizedBox(height: 8),
                          Wrap(
                            spacing: 6,
                            runSpacing: 6,
                            children: [
                              _MiniOrderChip(
                                text: '${order.items.length} item(s)',
                                icon: Icons.receipt_long_outlined,
                              ),
                              const _MiniOrderChip(
                                text: 'En local',
                                icon: Icons.storefront_outlined,
                              ),
                              _MiniOrderChip(
                                text: order.statusLabel ?? 'Pendiente',
                                icon: Icons.schedule_rounded,
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      TenantMoneyService.format(order.total, decimals: 2),
                      style: const TextStyle(
                        color: Color(0xFF059669),
                        fontSize: 15,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                ...order.items.map((item) {
                  final barber = item.barberUserName?.trim();
                  return Container(
                    margin: const EdgeInsets.only(bottom: 8),
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 9),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF8FAFC),
                      borderRadius: BorderRadius.circular(13),
                      border: Border.all(color: const Color(0xFFE2E8F0)),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(
                          item.isProduct
                              ? Icons.inventory_2_outlined
                              : Icons.content_cut_rounded,
                          size: 18,
                          color: const Color(0xFF475569),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                item.displayName,
                                maxLines: 2,
                                overflow: TextOverflow.ellipsis,
                                style: const TextStyle(
                                  color: Color(0xFF111827),
                                  fontSize: 13,
                                  fontWeight: FontWeight.w900,
                                ),
                              ),
                              const SizedBox(height: 3),
                              Text(
                                [
                                  'Cant. ${item.quantity}',
                                  if (barber != null && barber.isNotEmpty)
                                    '$professionalLabel: $barber',
                                ].join(' - '),
                                maxLines: 2,
                                overflow: TextOverflow.ellipsis,
                                style: const TextStyle(
                                  color: Color(0xFF667085),
                                  fontSize: 11.5,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          TenantMoneyService.format(
                            item.subtotal,
                            decimals: 2,
                          ),
                          style: const TextStyle(
                            color: Color(0xFF111827),
                            fontSize: 12.5,
                            fontWeight: FontWeight.w900,
                          ),
                        ),
                      ],
                    ),
                  );
                }),
                if (order.notes?.trim().isNotEmpty == true) ...[
                  const SizedBox(height: 4),
                  Text(
                    order.notes!.trim(),
                    style: const TextStyle(
                      color: Color(0xFF6B7280),
                      fontSize: 12.5,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: busy ? null : () => onReject(order),
                        icon: const Icon(Icons.close_rounded, size: 18),
                        label: const Text('Rechazar'),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: const Color(0xFFDC2626),
                          side: const BorderSide(color: Color(0xFFFECACA)),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: busy || !cashOpened
                            ? null
                            : () => onComplete(order),
                        icon: busy
                            ? const SizedBox(
                                width: 16,
                                height: 16,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white,
                                ),
                              )
                            : const Icon(Icons.check_rounded, size: 18),
                        label: Text(
                          cashOpened ? 'Atendido' : 'Abre caja',
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF111827),
                          foregroundColor: Colors.white,
                          disabledBackgroundColor: const Color(0xFFE5E7EB),
                          disabledForegroundColor: const Color(0xFF9CA3AF),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          );
        }),
      ],
    );
  }
}

class _LocalConsumptionPaymentDialog extends StatefulWidget {
  final double total;

  const _LocalConsumptionPaymentDialog({required this.total});

  @override
  State<_LocalConsumptionPaymentDialog> createState() =>
      _LocalConsumptionPaymentDialogState();
}

class _LocalConsumptionPaymentDialogState
    extends State<_LocalConsumptionPaymentDialog> {
  String _method = 'EFECTIVO';

  @override
  Widget build(BuildContext context) {
    const options = [
      ('EFECTIVO', 'Efectivo', Icons.payments_outlined),
      ('YAPE', 'Yape', Icons.phone_iphone_rounded),
      ('PLIN', 'Plin', Icons.phone_android_rounded),
      ('TARJETA', 'Tarjeta', Icons.credit_card_rounded),
    ];

    return AlertDialog(
      title: const Text('Finalizar consumo'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Total: ${TenantMoneyService.format(widget.total, decimals: 2)}',
            style: const TextStyle(
              fontWeight: FontWeight.w900,
              color: Color(0xFF111827),
            ),
          ),
          const SizedBox(height: 12),
          const Text(
            'Selecciona el metodo cobrado para registrarlo en caja.',
            style: TextStyle(
              color: Color(0xFF667085),
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 12),
          ...options.map((option) {
            final selected = _method == option.$1;
            return RadioListTile<String>(
              value: option.$1,
              groupValue: _method,
              dense: true,
              contentPadding: EdgeInsets.zero,
              onChanged: (value) {
                if (value != null) setState(() => _method = value);
              },
              secondary: Icon(
                option.$3,
                color: selected
                    ? const Color(0xFF111827)
                    : const Color(0xFF94A3B8),
              ),
              title: Text(
                option.$2,
                style: TextStyle(
                  fontWeight: selected ? FontWeight.w900 : FontWeight.w700,
                ),
              ),
            );
          }),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancelar'),
        ),
        ElevatedButton(
          onPressed: () => Navigator.pop(context, _method),
          child: const Text('Registrar venta'),
        ),
      ],
    );
  }
}

class _ProductOrdersSection extends StatelessWidget {
  final List<OwnerProductOrderModel> orders;
  final bool cashOpened;
  final int? processingOrderId;
  final ValueChanged<OwnerProductOrderModel> onApprove;
  final ValueChanged<OwnerProductOrderModel> onReject;
  final ValueChanged<OwnerProductOrderModel> onDeliver;

  const _ProductOrdersSection({
    required this.orders,
    required this.cashOpened,
    required this.processingOrderId,
    required this.onApprove,
    required this.onReject,
    required this.onDeliver,
  });

  bool _isPending(OwnerProductOrderModel order) =>
      order.status.trim().toUpperCase() == 'PENDING';

  bool _isApproved(OwnerProductOrderModel order) =>
      order.status.trim().toUpperCase() == 'APPROVED';

  String _statusText(OwnerProductOrderModel order) {
    switch (order.status.trim().toUpperCase()) {
      case 'APPROVED':
        return 'Aprobado';
      case 'DELIVERED':
        return 'Entregado';
      case 'REJECTED':
        return 'Rechazado';
      default:
        return 'Pendiente';
    }
  }

  String _paymentText(String method) {
    switch (method.trim().toUpperCase()) {
      case 'PAY_AT_SHOP':
        return 'Paga en caja';
      case 'YAPE':
        return 'Yape';
      case 'PLIN':
        return 'Plin';
      case 'CASH':
      case 'EFECTIVO':
        return 'Efectivo';
      default:
        return method.trim().isEmpty ? 'Sin metodo' : method.trim();
    }
  }

  @override
  Widget build(BuildContext context) {
    if (orders.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: const Color(0xFFFFF5D6),
                borderRadius: BorderRadius.circular(13),
              ),
              child: const Icon(Icons.shopping_bag_outlined,
                  color: Color(0xFFD4A017)),
            ),
            const SizedBox(width: 10),
            const Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Productos separados',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w900,
                      color: Color(0xFF17131E),
                    ),
                  ),
                  SizedBox(height: 2),
                  Text(
                    'Pedidos de clientes para aprobar y entregar en caja.',
                    style: TextStyle(
                      color: Color(0xFF6B7280),
                      fontSize: 12.5,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        ...orders.map((order) {
          final busy = processingOrderId == order.id;
          final imageUrl = order.productImageUrl?.trim() ?? '';

          return Container(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: const Color(0xFFEEDFB9)),
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
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    ClipRRect(
                      borderRadius: BorderRadius.circular(14),
                      child: imageUrl.isEmpty
                          ? Container(
                              width: 52,
                              height: 52,
                              color: const Color(0xFFF6F0E2),
                              child: const Icon(Icons.inventory_2_outlined,
                                  color: Color(0xFFB8872B)),
                            )
                          : Image.network(
                              imageUrl,
                              width: 52,
                              height: 52,
                              fit: BoxFit.cover,
                              errorBuilder: (_, __, ___) => Container(
                                width: 52,
                                height: 52,
                                color: const Color(0xFFF6F0E2),
                                child: const Icon(Icons.inventory_2_outlined,
                                    color: Color(0xFFB8872B)),
                              ),
                            ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            order.productName,
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              fontSize: 14.5,
                              fontWeight: FontWeight.w900,
                              color: Color(0xFF111827),
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '${order.customerName} ${order.customerPhone?.trim().isNotEmpty == true ? '• ${order.customerPhone}' : ''}',
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              color: Color(0xFF667085),
                              fontSize: 12.5,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Wrap(
                            spacing: 6,
                            runSpacing: 6,
                            children: [
                              _MiniOrderChip(
                                  text: 'Cant. ${order.quantity}',
                                  icon: Icons.tag_rounded),
                              _MiniOrderChip(
                                  text: _paymentText(order.paymentMethod),
                                  icon: Icons.payments_outlined),
                              _MiniOrderChip(
                                  text: _statusText(order),
                                  icon: _isApproved(order)
                                      ? Icons.check_circle_outline
                                      : Icons.schedule_rounded),
                            ],
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      TenantMoneyService.format(order.total, decimals: 2),
                      style: const TextStyle(
                        color: Color(0xFF059669),
                        fontSize: 15,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                  ],
                ),
                if (order.notes?.trim().isNotEmpty == true) ...[
                  const SizedBox(height: 10),
                  Text(
                    order.notes!.trim(),
                    style: const TextStyle(
                      color: Color(0xFF6B7280),
                      fontSize: 12.5,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
                const SizedBox(height: 12),
                if (_isPending(order))
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: busy ? null : () => onReject(order),
                          icon: const Icon(Icons.close_rounded, size: 18),
                          label: const Text('Rechazar'),
                          style: OutlinedButton.styleFrom(
                            foregroundColor: const Color(0xFFDC2626),
                            side: const BorderSide(color: Color(0xFFFECACA)),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(14),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: busy ? null : () => onApprove(order),
                          icon: busy
                              ? const SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white,
                                  ),
                                )
                              : const Icon(Icons.check_rounded, size: 18),
                          label: const Text('Aprobar'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF10B981),
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(14),
                            ),
                          ),
                        ),
                      ),
                    ],
                  )
                else if (_isApproved(order))
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed:
                          busy || !cashOpened ? null : () => onDeliver(order),
                      icon: busy
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: Colors.white,
                              ),
                            )
                          : const Icon(Icons.inventory_outlined, size: 18),
                      label: Text(
                        cashOpened
                            ? 'Entregar en caja'
                            : 'Abre caja para entregar',
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF111827),
                        foregroundColor: Colors.white,
                        disabledBackgroundColor: const Color(0xFFE5E7EB),
                        disabledForegroundColor: const Color(0xFF9CA3AF),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14),
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          );
        }),
      ],
    );
  }
}

class _MiniOrderChip extends StatelessWidget {
  final String text;
  final IconData icon;

  const _MiniOrderChip({required this.text, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F3EF),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: const Color(0xFFE9DFC8)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 13, color: const Color(0xFF7C6B51)),
          const SizedBox(width: 4),
          Text(
            text,
            style: const TextStyle(
              color: Color(0xFF5F5344),
              fontSize: 11.5,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }
}

class _SalesSection extends StatelessWidget {
  final List<CashSaleModel> sales;
  final bool canManageSales;
  final ValueChanged<CashSaleModel> onEditSale;
  final ValueChanged<CashSaleModel> onDeleteSale;

  const _SalesSection(
      {required this.sales,
      required this.canManageSales,
      required this.onEditSale,
      required this.onDeleteSale});

  Color _paymentColor(String method) {
    switch (method.trim().toUpperCase()) {
      case 'EFECTIVO':
        return const Color(0xFF16A34A);
      case 'YAPE':
        return const Color(0xFF7C3AED);
      case 'PLIN':
        return const Color(0xFF2563EB);
      case 'FREE':
      case 'GRATIS':
        return const Color(0xFF8A6A00);
      case 'TARJETA':
      case 'CARD':
        return const Color(0xFF0F172A);
      default:
        return const Color(0xFF667085);
    }
  }

  String _readBarberName(CashSaleModel sale) {
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

    return '${_ownerLabels.professionalSingular} no registrado';
  }

  String _saleDateText(CashSaleModel sale) {
    dynamic rawValue;

    try {
      final dynamic raw = sale;
      rawValue = raw.fechaCreacion;
    } catch (_) {}

    if (rawValue == null) {
      try {
        final dynamic raw = sale;
        rawValue = raw.saleDate;
      } catch (_) {}
    }

    if (rawValue == null) {
      try {
        final dynamic raw = sale;
        rawValue = raw.createdAt;
      } catch (_) {}
    }

    if (rawValue == null) return 'Fecha no disponible';

    DateTime? date;
    if (rawValue is DateTime) {
      date = rawValue;
    } else {
      date = DateTime.tryParse(rawValue.toString());
    }

    if (date == null) return rawValue.toString();

    final dd = date.day.toString().padLeft(2, '0');
    final mm = date.month.toString().padLeft(2, '0');
    final yyyy = date.year.toString();
    final hh = date.hour.toString().padLeft(2, '0');
    final min = date.minute.toString().padLeft(2, '0');

    return '$dd/$mm/$yyyy · $hh:$min';
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Ventas de hoy',
            style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w900,
                color: Color(0xFF17131E),
                letterSpacing: 0.1)),
        const SizedBox(height: 12),
        if (sales.isEmpty)
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: const Color(0xFFF0E7D3)),
                boxShadow: [
                  BoxShadow(
                      color: Colors.black.withOpacity(0.04),
                      blurRadius: 12,
                      offset: const Offset(0, 5))
                ]),
            child: const Row(children: [
              Icon(Icons.receipt_long_outlined, color: Color(0xFFB4ACB9)),
              SizedBox(width: 10),
              Expanded(
                  child: Text('Aún no hay ventas registradas hoy.',
                      style: TextStyle(
                          color: Color(0xFF6E6676),
                          fontSize: 13.5,
                          fontWeight: FontWeight.w600)))
            ]),
          )
        else
          ...sales.map((sale) {
            final customerName = (sale.customerName != null &&
                    sale.customerName!.trim().isNotEmpty)
                ? sale.customerName!.trim()
                : 'Cliente ocasional';
            final paymentMethod = sale.metodoPago.trim().isEmpty
                ? 'Sin método'
                : sale.metodoPago.trim();
            final barberName = _readBarberName(sale);
            final saleDateText = _saleDateText(sale);
            final paymentSummary = _salePaymentSummary(sale);
            final isCourtesy = normalizePaymentCode(sale.metodoPago) == 'FREE';

            return InkWell(
              borderRadius: BorderRadius.circular(18),
              onTap: () => _showSaleDetail(context, sale),
              child: Container(
                margin: const EdgeInsets.only(bottom: 12),
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(18),
                    border: Border.all(color: const Color(0xFFF0E9D8)),
                    boxShadow: [
                      BoxShadow(
                          color: Colors.black.withOpacity(0.045),
                          blurRadius: 12,
                          offset: const Offset(0, 5))
                    ]),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                        width: 46,
                        height: 46,
                        decoration: BoxDecoration(
                            color: isCourtesy
                                ? const Color(0xFFFFF3C4)
                                : const Color(0xFFF6E9B8),
                            borderRadius: BorderRadius.circular(14)),
                        child: Icon(
                            isCourtesy
                                ? Icons.redeem_rounded
                                : Icons.receipt_long_rounded,
                            color: const Color(0xFF3B2F14),
                            size: 22)),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(customerName,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                  fontSize: 15,
                                  fontWeight: FontWeight.w900,
                                  color: Color(0xFF17131E),
                                  height: 1.1)),
                          const SizedBox(height: 4),
                          Row(children: [
                            const Icon(Icons.content_cut_rounded,
                                size: 14, color: Color(0xFF8B7A4E)),
                            const SizedBox(width: 4),
                            Expanded(
                                child: Text(barberName,
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                    style: const TextStyle(
                                        fontSize: 12.8,
                                        fontWeight: FontWeight.w700,
                                        color: Color(0xFF6B7280))))
                          ]),
                          const SizedBox(height: 6),
                          Wrap(spacing: 8, runSpacing: 6, children: [
                            _SaleInfoChip(
                                icon: Icons.payments_rounded,
                                text: paymentSummary,
                                color: _paymentColor(paymentMethod)),
                            _SaleInfoChip(
                                icon: Icons.shopping_bag_outlined,
                                text: '${sale.items.length} item(s)',
                                color: const Color(0xFF667085))
                          ]),
                          const SizedBox(height: 7),
                          Row(children: [
                            const Icon(Icons.schedule_rounded,
                                size: 14, color: Color(0xFF8B7A4E)),
                            const SizedBox(width: 5),
                            Expanded(
                                child: Text(saleDateText,
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                    style: const TextStyle(
                                        fontSize: 12.4,
                                        fontWeight: FontWeight.w800,
                                        color: Color(0xFF7A5A22))))
                          ]),
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        if (canManageSales)
                          PopupMenuButton<String>(
                            padding: EdgeInsets.zero,
                            tooltip: 'Acciones',
                            icon: const Icon(Icons.more_vert_rounded,
                                size: 18, color: Color(0xFF667085)),
                            onSelected: (value) {
                              if (value == 'view') {
                                _showSaleDetail(context, sale);
                              } else if (value == 'edit') {
                                onEditSale(sale);
                              } else if (value == 'delete') {
                                onDeleteSale(sale);
                              }
                            },
                            itemBuilder: (_) => const [
                              PopupMenuItem<String>(
                                  value: 'view',
                                  child: Row(children: [
                                    Icon(Icons.visibility_rounded,
                                        size: 18, color: Color(0xFF047857)),
                                    SizedBox(width: 8),
                                    Text('Ver detalle')
                                  ])),
                              PopupMenuItem<String>(
                                  value: 'edit',
                                  child: Row(children: [
                                    Icon(Icons.edit_rounded,
                                        size: 18, color: Color(0xFF17131E)),
                                    SizedBox(width: 8),
                                    Text('Editar')
                                  ])),
                              PopupMenuItem<String>(
                                  value: 'delete',
                                  child: Row(children: [
                                    Icon(Icons.delete_outline_rounded,
                                        size: 18, color: Color(0xFFC84C4C)),
                                    SizedBox(width: 8),
                                    Text('Eliminar')
                                  ])),
                            ],
                          ),
                        if (canManageSales) const SizedBox(height: 6),
                        Text(
                          isCourtesy
                              ? 'Cortesía'
                              : '${TenantMoneyService.format(sale.total, decimals: 2)}',
                          style: TextStyle(
                            fontWeight: FontWeight.w900,
                            fontSize: 16,
                            color: isCourtesy
                                ? const Color(0xFF8A6A00)
                                : const Color(0xFF16A34A),
                          ),
                        ),
                        if (isCourtesy)
                          Text(
                            'Ref. ${TenantMoneyService.format(sale.subtotal, decimals: 2)}',
                            style: const TextStyle(
                              color: Color(0xFF7A7282),
                              fontSize: 11.5,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                      ],
                    ),
                  ],
                ),
              ),
            );
          }),
      ],
    );
  }

  String _methodLabel(String method) {
    switch (normalizePaymentCode(method)) {
      case 'CASH':
        return 'Efectivo';
      case 'CARD':
        return 'Tarjeta';
      case 'TRANSFER':
        return 'Transferencia';
      case 'YAPE':
        return 'Yape';
      case 'PLIN':
        return 'Plin';
      case 'MIXED':
        return 'Pago mixto';
      case 'FREE':
        return 'Gratis / Cortesía';
      default:
        return method.trim().isEmpty ? 'Sin método' : method.trim();
    }
  }

  String _salePaymentSummary(CashSaleModel sale) {
    if (sale.payments.isEmpty) return _methodLabel(sale.metodoPago);
    return sale.payments
        .map((payment) =>
            '${_methodLabel(payment.method)} ${TenantMoneyService.format(payment.amount, decimals: 2)}')
        .join(' + ');
  }

  void _showSaleDetail(BuildContext context, CashSaleModel sale) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _SaleDetailSheet(
        sale: sale,
        barberName: _readBarberName(sale),
        saleDateText: _saleDateText(sale),
        paymentSummary: _salePaymentSummary(sale),
        methodLabel: _methodLabel,
      ),
    );
  }
}

class _SaleDetailSheet extends StatelessWidget {
  final CashSaleModel sale;
  final String barberName;
  final String saleDateText;
  final String paymentSummary;
  final String Function(String) methodLabel;

  const _SaleDetailSheet({
    required this.sale,
    required this.barberName,
    required this.saleDateText,
    required this.paymentSummary,
    required this.methodLabel,
  });

  String get _customerName {
    final value = sale.customerName?.trim();
    return value == null || value.isEmpty ? 'Cliente ocasional' : value;
  }

  bool get isCourtesy => normalizePaymentCode(sale.metodoPago) == 'FREE';

  Widget _infoRow(String label, String value, {Color? valueColor}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 7),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Text(
              label,
              style: const TextStyle(
                color: Color(0xFF7A7282),
                fontWeight: FontWeight.w800,
                fontSize: 13,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Flexible(
            flex: 2,
            child: Text(
              value,
              textAlign: TextAlign.right,
              style: TextStyle(
                color: valueColor ?? const Color(0xFF17131E),
                fontWeight: FontWeight.w900,
                fontSize: 13.5,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(String title, IconData icon) {
    return Row(
      children: [
        Icon(icon, size: 17, color: const Color(0xFFD4AF37)),
        const SizedBox(width: 7),
        Text(
          title,
          style: const TextStyle(
            color: Color(0xFF17131E),
            fontWeight: FontWeight.w900,
            fontSize: 15,
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      initialChildSize: 0.82,
      minChildSize: 0.45,
      maxChildSize: 0.94,
      builder: (context, controller) {
        return Container(
          decoration: const BoxDecoration(
            color: Color(0xFFF6F3ED),
            borderRadius: BorderRadius.vertical(top: Radius.circular(30)),
          ),
          child: ListView(
            controller: controller,
            padding: const EdgeInsets.fromLTRB(18, 12, 18, 24),
            children: [
              Center(
                child: Container(
                  width: 42,
                  height: 5,
                  decoration: BoxDecoration(
                    color: const Color(0xFFD8D2C7),
                    borderRadius: BorderRadius.circular(99),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              ElectronicDocumentCard(saleId: sale.saleId, showEmpty: true),
              const SizedBox(height: 16),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    width: 52,
                    height: 52,
                    decoration: BoxDecoration(
                      color: const Color(0xFFF6E9B8),
                      borderRadius: BorderRadius.circular(18),
                    ),
                    child: const Icon(Icons.receipt_long_rounded,
                        color: Color(0xFF3B2F14)),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Detalle de venta',
                          style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w900,
                              color: Color(0xFF17131E)),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Venta #${sale.saleId} · $saleDateText',
                          style: const TextStyle(
                              color: Color(0xFF7A7282),
                              fontWeight: FontWeight.w700),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.close_rounded),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(color: const Color(0xFFF0E7D3)),
                  boxShadow: [
                    BoxShadow(
                        color: Colors.black.withOpacity(0.04),
                        blurRadius: 14,
                        offset: const Offset(0, 6))
                  ],
                ),
                child: Column(
                  children: [
                    _infoRow('Cliente', _customerName),
                    _infoRow(_ownerLabels.professionalSingular, barberName),
                    _infoRow('Método', paymentSummary),
                    if (sale.appointmentId != null)
                      _infoRow('Cita', '#${sale.appointmentId}'),
                    _infoRow(
                      isCourtesy ? 'Cortesía' : 'Total',
                      isCourtesy
                          ? '${TenantMoneyService.format(0)} · Ref. ${TenantMoneyService.format(sale.subtotal, decimals: 2)}'
                          : '${TenantMoneyService.format(sale.total, decimals: 2)}',
                      valueColor: isCourtesy
                          ? const Color(0xFF8A6A00)
                          : const Color(0xFF047857),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(color: const Color(0xFFF0E7D3)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _sectionTitle(
                        'Servicios y productos', Icons.shopping_bag_outlined),
                    const SizedBox(height: 12),
                    if (sale.items.isEmpty)
                      const Text(
                        'Esta venta no tiene items detallados desde el backend.',
                        style: TextStyle(
                            color: Color(0xFF7A7282),
                            fontWeight: FontWeight.w700),
                      )
                    else
                      ...sale.items.map((item) {
                        final name =
                            (item.serviceName?.trim().isNotEmpty == true)
                                ? item.serviceName!.trim()
                                : (item.productName?.trim().isNotEmpty == true)
                                    ? item.productName!.trim()
                                    : 'Item registrado';
                        final barber = item.barberUserName?.trim();
                        return Container(
                          margin: const EdgeInsets.only(bottom: 10),
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: const Color(0xFFFAF8F3),
                            borderRadius: BorderRadius.circular(18),
                            border: Border.all(color: const Color(0xFFF0E7D3)),
                          ),
                          child: Row(
                            children: [
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(name,
                                        style: const TextStyle(
                                            fontWeight: FontWeight.w900,
                                            color: Color(0xFF17131E))),
                                    const SizedBox(height: 4),
                                    Text(
                                      'Cant. ${item.cantidad} · Unit. ${TenantMoneyService.format(item.precioUnitario, decimals: 2)}${barber != null && barber.isNotEmpty ? ' · $barber' : ''}',
                                      style: const TextStyle(
                                          color: Color(0xFF7A7282),
                                          fontWeight: FontWeight.w700,
                                          fontSize: 12.5),
                                    ),
                                  ],
                                ),
                              ),
                              Text(
                                '${TenantMoneyService.format(item.subtotal, decimals: 2)}',
                                style: const TextStyle(
                                    fontWeight: FontWeight.w900,
                                    color: Color(0xFF17131E)),
                              ),
                            ],
                          ),
                        );
                      }),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(color: const Color(0xFFF0E7D3)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _sectionTitle('Pagos y resumen', Icons.payments_rounded),
                    const SizedBox(height: 12),
                    if (sale.payments.isNotEmpty) ...[
                      ...sale.payments.map((payment) => _infoRow(
                          methodLabel(payment.method),
                          '${TenantMoneyService.format(payment.amount, decimals: 2)}')),
                      const Divider(height: 18),
                    ],
                    _infoRow('Subtotal',
                        '${TenantMoneyService.format(sale.subtotal, decimals: 2)}'),
                    _infoRow('Descuento',
                        '${TenantMoneyService.format(sale.discount, decimals: 2)}'),
                    _infoRow('Recibido',
                        '${TenantMoneyService.format(sale.cashReceived, decimals: 2)}'),
                    _infoRow('Vuelto',
                        '${TenantMoneyService.format(sale.changeAmount, decimals: 2)}'),
                    const Divider(height: 18),
                    _infoRow(
                      isCourtesy ? 'Total final cortesía' : 'Total final',
                      isCourtesy
                          ? '${TenantMoneyService.format(0)}'
                          : '${TenantMoneyService.format(sale.total, decimals: 2)}',
                      valueColor: isCourtesy
                          ? const Color(0xFF8A6A00)
                          : const Color(0xFF047857),
                    ),
                    if (sale.isPendingValidation && sale.puntosPendientes > 0)
                      _infoRow('Puntos pendientes al aprobar',
                          '${sale.puntosPendientes}'),
                    if (!sale.isPendingValidation && sale.puntosGanados > 0)
                      _infoRow('Puntos ganados', '${sale.puntosGanados}'),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _SaleInfoChip extends StatelessWidget {
  final IconData icon;
  final String text;
  final Color color;

  const _SaleInfoChip({
    required this.icon,
    required this.text,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
      decoration: BoxDecoration(
        color: color.withOpacity(0.09),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withOpacity(0.16)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 13, color: color),
          const SizedBox(width: 4),
          Flexible(
            child: Text(
              text,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              softWrap: false,
              style: TextStyle(
                color: color,
                fontSize: 11.5,
                fontWeight: FontWeight.w900,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _EmptyCashSection extends StatelessWidget {
  final VoidCallback onOpen;
  final String branchName;

  const _EmptyCashSection({required this.onOpen, required this.branchName});

  @override
  Widget build(BuildContext context) => SuperGodsStateCard(
        icon: Icons.storefront_rounded,
        title: context.ownerCoreCopy.text('cashReady'),
        message:
            context.ownerCoreCopy.withValue('cashReadyDescription', branchName),
        actionLabel: context.ownerCoreCopy.text('openCash'),
        onAction: onOpen,
      );
}

class _ErrorCard extends StatelessWidget {
  final String message;
  final VoidCallback onRetry;

  const _ErrorCard({required this.message, required this.onRetry});

  @override
  Widget build(BuildContext context) => SuperGodsStateCard(
        icon: Icons.cloud_off_rounded,
        title: context.ownerCoreCopy.text('cashLoadError'),
        message: SuperGodsUi.humanError(message),
        actionLabel: context.ownerCoreCopy.text('retryShort'),
        onAction: onRetry,
        error: true,
      );
}

class _MiniCard extends StatelessWidget {
  final String title;
  final String value;
  final Color? valueColor;
  final bool highlighted;

  const _MiniCard({
    required this.title,
    required this.value,
    this.valueColor,
    this.highlighted = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: highlighted ? const Color(0xFFFFFBEB) : Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color:
              highlighted ? const Color(0xFFFDE68A) : const Color(0xFFF0E7D3),
        ),
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
          Text(
            title,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              fontSize: 12.5,
              fontWeight: FontWeight.w700,
              color: Color(0xFF7A7282),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            value,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w900,
              color: valueColor ?? const Color(0xFF17131E),
            ),
          ),
        ],
      ),
    );
  }
}
