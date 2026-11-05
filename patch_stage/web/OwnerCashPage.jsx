import { useEffect, useMemo, useRef, useState } from 'react';
import { PremiumEmptyState, PremiumErrorState, premiumConfirm } from '../../components/PremiumUi';
import {
  approveCashMovement,
  approveSalePayment,
  closeCashRegister,
  createBarberPayment,
  createCashFundMovement,
  updateCashFundMovement,
  deleteCashFundMovement,
  createCashMovement,
  createCashSale,
  deleteCashMovement,
  deleteCashSale,
  getBarberPaymentPreview,
  getCashBarbers,
  getCashAudit,
  getCashHistory,
  getCashFundSummary,
  getCashFundMovements,
  getCashFundMovementSummary,
  getCashProducts,
  getCashServices,
  getCashMovements,
  getCashPayableEmployees,
  getCurrentCashRegister,
  getPendingCashReconciliation,
  getOwnerBranches,
  getOwnerPaymentMethods,
  getPendingValidationSales,
  getSalesByCashRegister,
  getTodayCashSales,
  openCashRegister,
  rejectCashMovement,
  reconcileCashRegister,
  rejectSalePayment,
  updateCashMovement,
  updateCashSale,
} from '../../api/ownerCashApi';
import { createOwnerCustomer, getOwnerCustomers } from '../../api/ownerCustomersApi';
import { getBarberServiceAssignment } from '../../api/ownerBarbersApi';
import {
  approveOwnerProductOrder,
  deliverOwnerProductOrder,
  getOwnerProductOrders,
  rejectOwnerProductOrder,
} from '../../api/ownerProductOrdersApi';
import { getMyOwnerPermissions } from '../../api/ownerPermissionsApi';
import { useAuth } from '../../context/AuthContext';
import { getBusinessLabels, readBusinessLabels } from '../../utils/businessLabels';
import { hasAnyOwnerPermission } from '../../utils/ownerPermissions';
import { formatTenantMoney, getTenantCurrencySymbol } from '../../utils/tenantMoney';
import { exportCashHistoryExcel, exportCashHistoryPdf } from '../../utils/cashHistoryExport';
import { autoPrintApprovedSale, printElectronicPdfOnThermal } from '../../services/qzPrinterService';
import { whatsappPhoneDigits } from '../../utils/internationalPhone';
import InternationalPhoneField from '../../components/InternationalPhoneField';
import { downloadBase64File, getElectronicDocumentFiles, getElectronicInvoicingAccess, getSaleElectronicDocuments, issueElectronicDocument, refreshElectronicDocument, retryElectronicDocument } from '../../api/electronicInvoicingApi';

function useAllowedServicesForBarber({ barberId, branchId, setSelectedServiceId }) {
  const [allowedServiceIds, setAllowedServiceIds] = useState(null);
  const [loadingAllowedServices, setLoadingAllowedServices] = useState(false);

  useEffect(() => {
    let active = true;
    if (!barberId || !branchId) {
      setAllowedServiceIds(null);
      setLoadingAllowedServices(false);
      return () => { active = false; };
    }

    setLoadingAllowedServices(true);
    setAllowedServiceIds([]);
    getBarberServiceAssignment({ barberId: Number(barberId), branchId: Number(branchId) })
      .then((assignment) => {
        if (!active) return;
        const next = assignment?.configured
          ? (assignment.serviceIds || []).map(Number)
          : null;
        setAllowedServiceIds(next);
        setLoadingAllowedServices(false);
        if (Array.isArray(next)) {
          setSelectedServiceId((current) =>
            current && !next.includes(Number(current)) ? '' : current
          );
        }
      })
      .catch(() => {
        if (active) {
          setAllowedServiceIds(null);
          setLoadingAllowedServices(false);
        }
      });

    return () => { active = false; };
  }, [barberId, branchId, setSelectedServiceId]);

  return { allowedServiceIds, loadingAllowedServices };
}

function filterServicesForBarber(services, allowedServiceIds) {
  if (!Array.isArray(allowedServiceIds)) return services;
  const allowed = new Set(allowedServiceIds.map(Number));
  return services.filter((service) => allowed.has(Number(service.id)));
}

function formatMoney(value) {
  return formatTenantMoney(value);
}

function formatDateTime(value) {
  if (!value) return '-';

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString('es-PE', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function normalizeMethod(value) {
  const code = String(value || '').trim().toUpperCase();

  if (code === 'EFECTIVO') return 'CASH';
  if (code === 'TARJETA') return 'CARD';
  if (code === 'TRANSFERENCIA') return 'TRANSFER';

  return code;
}

function methodLabel(value) {
  const code = normalizeMethod(value);

  const labels = {
    CASH: 'Efectivo',
    YAPE: 'Yape',
    PLIN: 'Plin',
    CARD: 'Tarjeta',
    TRANSFER: 'Transferencia',
    MIXED: 'Pago mixto',
    DEPOSIT_APPLIED: 'Inicial aplicado',
    FREE: 'Gratis',
    NEQUI: 'Nequi',
    DAVIPLATA: 'Daviplata',
    PAGO_MOVIL: 'Pago móvil',
    ZELLE: 'Zelle',
    QR: 'QR',
  };

  return labels[code] || code || 'Otro';
}

function ownerAsBarberOption(session) {
  const role = String(session?.role || '').trim().toUpperCase();
  const id = Number(session?.userId || 0);

  if (role !== 'OWNER' || !id) return null;

  const name = String(session?.userName || '').trim();

  return {
    id,
    name: name ? `${name} (Dueño)` : 'Dueño del negocio',
    owner: true,
  };
}

function mergeOwnerIntoBarbers(barbers, session) {
  const cleanBarbers = Array.isArray(barbers)
    ? barbers.filter((item) => Number(item?.id || 0) > 0)
    : [];
  const ownerOption = ownerAsBarberOption(session);

  if (!ownerOption) return cleanBarbers;

  const exists = cleanBarbers.some((item) => String(item.id) === String(ownerOption.id));
  if (exists) return cleanBarbers;

  return [ownerOption, ...cleanBarbers];
}

function parseMoneyInput(value) {
  const parsed = Number(String(value ?? '').replace(',', '.'));
  return Number.isNaN(parsed) ? 0 : parsed;
}

function roundMoney(value) {
  return Math.round((Number(value || 0) + Number.EPSILON) * 100) / 100;
}

function createPaymentDraft(method = 'CASH', amount = 0) {
  return {
    key: `payment-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    method,
    amount: Number(amount || 0).toFixed(2),
  };
}

function salePaymentsOf(sale) {
  const raw = Array.isArray(sale?.payments)
    ? sale.payments
    : Array.isArray(sale?.salePayments)
      ? sale.salePayments
      : [];

  return raw
    .map((payment) => ({
      method: payment?.method ?? payment?.paymentMethod ?? payment?.metodoPago,
      amount: Number(payment?.amount ?? payment?.totalAmount ?? 0),
    }))
    .filter((payment) => payment.method && payment.amount > 0);
}

function salePaymentSummary(sale) {
  const payments = salePaymentsOf(sale);

  if (payments.length === 0) {
    return methodLabel(sale?.metodoPago);
  }

  return payments
    .map((payment) => `${methodLabel(payment.method)} ${formatMoney(payment.amount)}`)
    .join(' + ');
}

function customerWhatsappUrlOf(sale) {
  const provided = String(
    sale?.customerWhatsappUrl ??
      sale?.whatsappUrl ??
      sale?.customerWhatsappLink ??
      ''
  ).trim();

  if (provided) return provided;

  const phone = normalizeWhatsappPhone(
    sale?.customerPhone ??
      sale?.telefono ??
      sale?.phone ??
      sale?.customerTelefono ??
      sale?.whatsappPhone ??
      sale?.customer?.telefono ??
      sale?.customer?.phone ??
      ''
  );

  if (!phone) return '';

  const message = customerWhatsappMessageOf(sale);
  if (!message) return '';

  const query = message ? `?text=${encodeURIComponent(message)}` : '';
  return `https://wa.me/${phone}${query}`;
}

function customerWhatsappMessageOf(sale) {
  return String(
    sale?.customerWhatsappMessage ??
      sale?.whatsappMessage ??
      sale?.message ??
      ''
  ).trim();
}

function saleWithWhatsappFallback(primary, fallback) {
  const merged = { ...(fallback || {}), ...(primary || {}) };

  for (const key of [
    'customerWhatsappUrl',
    'whatsappUrl',
    'customerWhatsappLink',
    'customerWhatsappMessage',
    'whatsappMessage',
    'message',
    'customerPhone',
    'telefono',
    'phone',
    'customerTelefono',
    'whatsappPhone',
    'customerName',
  ]) {
    const value = String(primary?.[key] ?? '').trim();
    if (!value && fallback?.[key] !== undefined) {
      merged[key] = fallback[key];
    }
  }

  return merged;
}

function normalizeWhatsappPhone(value) {
  return whatsappPhoneDigits(value);
}

async function offerCustomerWhatsappFollowUp(sale, { canOpenWhatsapp = false } = {}) {
  if (!canOpenWhatsapp) return;

  const url = customerWhatsappUrlOf(sale);
  if (!url) return;

  const customer = String(
    sale?.customerName ||
      sale?.customer?.fullName ||
      sale?.customer?.nombre ||
      'cliente'
  ).trim();
  const message = customerWhatsappMessageOf(sale);
  const preview = message ? `\n\nMensaje:\n${message}` : '';
  const shouldOpen = await premiumConfirm(
    `Venta registrada. ¿Enviar WhatsApp a ${customer}?${preview}`
  );

  if (!shouldOpen) return;

  const opened = window.open(url, '_blank', 'noopener,noreferrer');
  if (!opened) {
    window.prompt('Copia este enlace para abrir WhatsApp:', url);
  }
}

const DEFAULT_PAYMENT_METHODS = [
  { code: 'CASH', label: 'Efectivo', icon: '💵', helper: 'Efectivo físico' },
  { code: 'YAPE', label: 'Yape', icon: '🟣', helper: 'Pago digital' },
  { code: 'PLIN', label: 'Plin', icon: '🔵', helper: 'Pago digital' },
  { code: 'CARD', label: 'Tarjeta', icon: '💳', helper: 'Tarjeta / POS' },
  { code: 'TRANSFER', label: 'Transferencia', icon: '🏦', helper: 'Transferencias' },
];

const BASE_PAYMENT_METHODS = [
  { code: 'CASH', label: 'Efectivo', icon: '💵', helper: 'Efectivo físico' },
  { code: 'CARD', label: 'Tarjeta', icon: '💳', helper: 'Tarjeta / POS' },
  { code: 'TRANSFER', label: 'Transferencia', icon: '🏦', helper: 'Transferencias' },
];

const METHOD_ICON_BY_CODE = {
  CASH: '💵',
  YAPE: '🟣',
  PLIN: '🔵',
  CARD: '💳',
  TRANSFER: '🏦',
  NEQUI: '🟢',
  DAVIPLATA: '🔴',
  PAGO_MOVIL: '📲',
  ZELLE: '💸',
  QR: '▣',
};

function normalizePaymentMethodOptions(methods = []) {
  const source = Array.isArray(methods) && methods.length > 0 ? methods : DEFAULT_PAYMENT_METHODS;
  const map = new Map();

  source.forEach((item) => {
    const code = normalizeMethod(item?.code ?? item?.method ?? item?.paymentMethod ?? item?.value);
    if (!code || code === 'MIXED' || code === 'FREE' || code === 'DEPOSIT_APPLIED') return;

    const label = String(item?.displayName ?? item?.label ?? item?.name ?? methodLabel(code)).trim();
    map.set(code, {
      code,
      label: label || methodLabel(code),
      icon: item?.icon || METHOD_ICON_BY_CODE[code] || '•',
      helper: item?.helper || (code === 'CASH' ? 'Efectivo físico' : 'Método configurado'),
    });
  });

  if (map.size === 0) {
    DEFAULT_PAYMENT_METHODS.forEach((method) => map.set(method.code, method));
  }

  return Array.from(map.values());
}

function paymentSelectOptions(methods = []) {
  return normalizePaymentMethodOptions(methods).map((method) => ({
    value: method.code,
    label: method.label,
  }));
}

function balanceRowsForSource(cashRegister, fundingSource) {
  const key = fundingSource === 'ACCUMULATED_FUND'
    ? 'accumulatedFundBalances'
    : 'paymentMethodBalances';
  return Array.isArray(cashRegister?.[key]) ? cashRegister[key] : [];
}

function availableBalanceForMethod(cashRegister, fundingSource, paymentMethod) {
  const method = normalizeMethod(paymentMethod);
  const row = balanceRowsForSource(cashRegister, fundingSource)
    .find((item) => summaryMethodOf(item) === method);

  if (row) return summaryAmountOf(row);
  if (fundingSource === 'CASH_REGISTER' && method === 'CASH') {
    return Number(cashRegister?.closingAmountExpected || 0);
  }
  return 0;
}

function totalBalanceForSource(cashRegister, fundingSource) {
  if (fundingSource === 'ACCUMULATED_FUND') {
    const total = Number(cashRegister?.accumulatedFundBalance);
    if (Number.isFinite(total)) return total;
  }

  const rows = balanceRowsForSource(cashRegister, fundingSource);
  if (rows.length > 0) {
    return rows.reduce((sum, item) => sum + summaryAmountOf(item), 0);
  }

  return fundingSource === 'CASH_REGISTER'
    ? Number(cashRegister?.closingAmountExpected || 0)
    : 0;
}

function fundingSourceSelectOptions(cashRegister, paymentMethod = null, canManageFund = false) {
  const method = normalizeMethod(paymentMethod);
  const balanceLabel = (source) => method
    ? `${methodLabel(method)} disponible ${formatMoney(availableBalanceForMethod(cashRegister, source, method))}`
    : `disponible total ${formatMoney(totalBalanceForSource(cashRegister, source))}`;

  return [
    { value: 'CASH_REGISTER', label: `Caja actual · ${balanceLabel('CASH_REGISTER')}` },
    { value: 'ACCUMULATED_FUND', label: `Fondo acumulado · ${balanceLabel('ACCUMULATED_FUND')}` },
    { value: 'EXTERNAL', label: 'Cuenta digital o externa · sin afectar saldos internos' },
  ].filter((option) => canManageFund || option.value !== 'ACCUMULATED_FUND');
}

function paymentSelectOptionsWithBalances(methods, cashRegister, fundingSource) {
  const options = paymentSelectOptions(methods);
  if (fundingSource === 'EXTERNAL') return options;

  return options.map((option) => ({
    ...option,
    label: `${option.label} · disponible ${formatMoney(
      availableBalanceForMethod(cashRegister, fundingSource, option.value)
    )}`,
  }));
}

function paymentLabelFromOptions(methods = [], code) {
  const normalized = normalizeMethod(code);
  return normalizePaymentMethodOptions(methods).find((method) => method.code === normalized)?.label || methodLabel(normalized);
}

function defaultExtraPaymentMethod(methods = []) {
  const options = normalizePaymentMethodOptions(methods);
  return options.find((method) => method.code !== 'CASH')?.code || 'CARD';
}

function summaryMethodOf(item) {
  return normalizeMethod(item?.paymentMethod ?? item?.method ?? item?.metodoPago ?? item?.code);
}

function summaryAmountOf(item) {
  return Number(item?.totalAmount ?? item?.amount ?? item?.total ?? 0);
}

function summaryCountOf(item) {
  return Number(item?.count ?? item?.quantity ?? item?.operations ?? 0);
}

function buildPaymentMethodRows(salesSummary = [], balanceSummary = [], configuredMethods = DEFAULT_PAYMENT_METHODS) {
  const codes = [];

  function addCode(code) {
    const normalized = normalizeMethod(code);
    if (!normalized || normalized === 'FREE' || normalized === 'DEPOSIT_APPLIED') return;
    if (!codes.includes(normalized)) codes.push(normalized);
  }

  const methodOptions = normalizePaymentMethodOptions(configuredMethods);
  methodOptions.forEach((method) => addCode(method.code));
  salesSummary.forEach((item) => addCode(summaryMethodOf(item)));
  balanceSummary.forEach((item) => addCode(summaryMethodOf(item)));

  return codes.map((code) => {
    const base = methodOptions.find((method) => method.code === code);
    const salesItem = salesSummary.find((item) => summaryMethodOf(item) === code);
    const balanceItem = balanceSummary.find((item) => summaryMethodOf(item) === code);

    return {
      code,
      label: base?.label || methodLabel(code),
      icon: base?.icon || '•',
      helper: base?.helper || 'Método configurado',
      salesAmount: summaryAmountOf(salesItem),
      balanceAmount: balanceItem ? summaryAmountOf(balanceItem) : summaryAmountOf(salesItem),
      count: summaryCountOf(salesItem || balanceItem),
    };
  });
}

function PaymentMethodCard({ method }) {
  const isCash = method.code === 'CASH';
  const balance = Number(method.balanceAmount || 0);
  const sales = Number(method.salesAmount || 0);

  return (
    <div className={`rounded-[26px] border p-5 shadow-[0_12px_32px_rgba(15,23,42,0.045)] ${
      isCash
        ? 'border-emerald-200 bg-emerald-50'
        : 'border-neutral-200 bg-white'
    }`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs font-black uppercase tracking-[0.16em] text-neutral-500">
            {method.label || methodLabel(method.code)}
          </div>
          <div className={`mt-2 text-2xl font-black ${amountClassByValue(balance)}`}>
            {formatMoney(balance)}
          </div>
        </div>

        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white text-xl shadow-sm">
          {method.icon}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 text-xs font-bold">
        <div className="rounded-2xl bg-white/80 px-3 py-2">
          <div className="text-neutral-400">Vendido</div>
          <div className="mt-1 text-neutral-950">{formatMoney(sales)}</div>
        </div>
        <div className="rounded-2xl bg-white/80 px-3 py-2">
          <div className="text-neutral-400">Operaciones</div>
          <div className="mt-1 text-neutral-950">{method.count || 0}</div>
        </div>
      </div>

      <div className="mt-3 text-xs font-semibold text-neutral-500">
        {isCash ? 'Saldo físico esperado con apertura y movimientos.' : method.helper}
      </div>
    </div>
  );
}

function movementTypeLabel(type) {
  const businessLabels = readBusinessLabels();
  const labels = {
    INCOME: 'Ingreso',
    EXPENSE: 'Gasto',
    ADVANCE_BARBER: 'Adelanto ' + businessLabels.professionalSingular,
    PAYMENT_BARBER: 'Pago ' + businessLabels.professionalSingular,
    PAYMENT_METHOD_TRANSFER: 'Traslado entre métodos',
  };

  return labels[type] || type || 'Movimiento';
}

function auditActionLabel(action) {
  const labels = {
    UPDATE: 'Edito',
    DELETE: 'Elimino',
  };

  return labels[action] || action || 'Cambio';
}

function auditEntityLabel(entityType) {
  const labels = {
    SALE: 'venta',
    CASH_MOVEMENT: 'movimiento',
  };

  return labels[entityType] || 'registro';
}

function saleValidationStatus(sale) {
  return String(
    sale?.paymentValidationStatus ??
      sale?.validationStatus ??
      sale?.estadoValidacionPago ??
      sale?.estadoPago ??
      ''
  )
    .trim()
    .toUpperCase();
}

function isPendingSaleValidation(sale) {
  const status = saleValidationStatus(sale);

  if (!status) return true;

  return (
    status === 'PENDING_VALIDATION' ||
    status === 'PENDING_APPROVAL' ||
    status === 'PENDING_REVIEW' ||
    status === 'PENDING' ||
    status === 'PENDIENTE' ||
    status === 'PENDIENTE_VALIDACION'
  );
}

function saleCreatedBy(sale) {
  return (
    sale?.createdByUserName ||
    sale?.createdByName ||
    sale?.createdBy ||
    sale?.barberName ||
    saleBarberName(sale)
  );
}

function movementStatusCode(movement) {
  return String(
    movement?.status ??
      movement?.approvalStatus ??
      movement?.estado ??
      movement?.estadoAprobacion ??
      ''
  )
    .trim()
    .toUpperCase();
}

function isPendingOwnerApproval(movement) {
  const code = movementStatusCode(movement);

  return (
    code === 'PENDING' ||
    code === 'PENDING_REVIEW' ||
    code === 'PENDING_APPROVAL' ||
    code === 'REQUESTED' ||
    code === 'SOLICITADO' ||
    code === 'PENDIENTE' ||
    code === 'PENDIENTE_APROBACION'
  );
}

function movementRequestedBy(movement) {
  return (
    movement?.requestedByUserName ||
    movement?.createdByUserName ||
    movement?.userName ||
    movement?.adminName ||
    'Usuario'
  );
}

function movementAmountClass(type) {
  if (type === 'INCOME') return 'text-emerald-700';
  if (type === 'PAYMENT_METHOD_TRANSFER') return 'text-blue-700';
  return 'text-red-700';
}

function amountClassByValue(value) {
    const number = Number(value || 0);
  
    if (number < 0) return 'text-red-700';
    if (number > 0) return 'text-neutral-950';
  
    return 'text-neutral-500';
  }
  
  function balanceTone(value) {
    const number = Number(value || 0);
  
    if (number < 0) return 'red';
    if (number > 0) return 'dark';
  
    return 'default';
  }

function StatCard({ title, value, helper, tone = 'default' }) {
  const styles = {
    default: 'border-neutral-200 bg-white',
    gold: 'border-amber-200 bg-[linear-gradient(135deg,#FFFBEB_0%,#FFFFFF_70%)]',
    dark: 'border-neutral-900 bg-neutral-950 text-white',
    green: 'border-emerald-200 bg-emerald-50',
    red: 'border-red-200 bg-red-50',
  };

  return (
    <div className={`rounded-[28px] border p-5 shadow-[0_12px_32px_rgba(15,23,42,0.045)] ${styles[tone]}`}>
      <div className={tone === 'dark' ? 'text-sm font-bold text-white/55' : 'text-sm font-bold text-neutral-500'}>
        {title}
      </div>

      <div className={tone === 'dark' ? 'mt-2 text-2xl font-black text-white' : 'mt-2 text-2xl font-black text-neutral-950'}>
        {value}
      </div>

      {helper && (
        <div className={tone === 'dark' ? 'mt-1 text-xs text-white/45' : 'mt-1 text-xs text-neutral-500'}>
          {helper}
        </div>
      )}
    </div>
  );
}


function toDateInputValue(date) {
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, '0');
  const dd = String(date.getDate()).padStart(2, '0');

  return `${yyyy}-${mm}-${dd}`;
}


function buildLocalSaleDateForBackend(dateValue) {
  const now = new Date();
  const thirtyDaysAgo = new Date(now);
  thirtyDaysAgo.setDate(now.getDate() - 30);

  const todayValue = toDateInputValue(now);
  const minValue = toDateInputValue(thirtyDaysAgo);
  let selected = String(dateValue || todayValue).trim();

  if (!/^\d{4}-\d{2}-\d{2}$/.test(selected)) {
    selected = todayValue;
  }

  if (selected > todayValue) selected = todayValue;
  if (selected < minValue) selected = minValue;

  // Usamos mediodía local para evitar que una conversión de zona horaria
  // haga que la venta caiga en el día anterior. Importante probar también
  // en móvil con tenants de Venezuela (America/Caracas) y Perú (America/Lima).
  return `${selected}T12:00:00`;
}


function appointmentNumber(...values) {
  for (const value of values) {
    const number = Number(value ?? 0);
    if (!Number.isNaN(number) && number > 0) return number;
  }
  return 0;
}

function normalizeDepositStatus(value) {
  return String(value || '').trim().toUpperCase();
}

function isDepositApproved(value) {
  const status = normalizeDepositStatus(value);
  return (
    status === 'PAID' ||
    status === 'VALIDADO' ||
    status === 'VALIDATED' ||
    status === 'APPROVED' ||
    status === 'APROBADO'
  );
}

function saleDateInputMinValue() {
  const date = new Date();
  date.setDate(date.getDate() - 30);
  return toDateInputValue(date);
}

function saleDateInputMaxValue() {
  return toDateInputValue(new Date());
}

function saleIdOf(sale) {
  return sale?.saleId ?? sale?.id ?? sale?.sale_id;
}

function saleDateOf(sale) {
  return (
    sale?.fechaCreacion ??
    sale?.saleDate ??
    sale?.createdAt ??
    sale?.created_at ??
    sale?.fecha ??
    null
  );
}

function saleBarberName(sale) {
  const labels = readBusinessLabels();
  const items = Array.isArray(sale?.items) ? sale.items : [];
  const itemBarbers = [];

  for (const item of items) {
    const candidate = String(
      item?.barberUserName ??
        item?.barberName ??
        item?.barbero ??
        ''
    ).trim();

    if (candidate && !itemBarbers.includes(candidate)) {
      itemBarbers.push(candidate);
    }
  }

  if (itemBarbers.length > 0) return itemBarbers.join(', ');

  const direct = String(
    sale?.barberName ??
      sale?.barbero ??
      sale?.barberUserName ??
      ''
  ).trim();

  if (direct) return direct;

  return `${labels.professionalSingular[0].toUpperCase()}${labels.professionalSingular.slice(1)} no registrado`;
}

function isCourtesySale(sale) {
  return normalizeMethod(sale?.metodoPago) === 'FREE';
}

function saleReferenceValue(sale) {
  const subtotal = Number(sale?.subtotal ?? 0);
  if (subtotal > 0) return subtotal;

  const itemsTotal = saleItemsOf(sale).reduce((sum, item) => sum + saleItemTotal(item), 0);
  if (itemsTotal > 0) return itemsTotal;

  return Number(sale?.discount ?? 0);
}

function buildCourtesySummary(sales = []) {
  const byBarber = new Map();
  let count = 0;
  let referenceValue = 0;

  sales.filter(isCourtesySale).forEach((sale) => {
    const items = saleItemsOf(sale);
    const serviceItems = items.filter((item) => {
      const hasService = item?.serviceId || String(item?.serviceName ?? '').trim();
      const hasProduct = item?.productId || String(item?.productName ?? '').trim();
      return hasService || !hasProduct;
    });

    const entries = serviceItems.length > 0
      ? serviceItems.map((item) => ({
          barberName: saleItemBarberName(item) === '-' ? saleBarberName(sale) : saleItemBarberName(item),
          count: saleItemQuantity(item),
          referenceValue: saleItemTotal(item) || saleItemUnitPrice(item),
        }))
      : [{
          barberName: saleBarberName(sale),
          count: 1,
          referenceValue: saleReferenceValue(sale),
        }];

    entries.forEach((entry) => {
      const itemCount = Math.max(1, Number(entry.count || 1));
      const itemValue = Number(entry.referenceValue || 0);
      const current = byBarber.get(entry.barberName) || {
        barberName: entry.barberName,
        count: 0,
        referenceValue: 0,
      };

      current.count += itemCount;
      current.referenceValue += itemValue;
      byBarber.set(entry.barberName, current);
      count += itemCount;
      referenceValue += itemValue;
    });
  });

  return {
    count,
    referenceValue,
    byBarber: Array.from(byBarber.values()).sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      return b.referenceValue - a.referenceValue;
    }),
  };
}

function readAttendAppointmentFromStorage() {
  try {
    const raw = window.sessionStorage.getItem('ownerWebAttendAppointment');
    if (!raw) return null;

    const parsed = JSON.parse(raw);
    if (!parsed?.appointmentId) return null;

    return parsed;
  } catch {
    return null;
  }
}

function clearAttendAppointmentFromStorage() {
  window.sessionStorage.removeItem('ownerWebAttendAppointment');
}

function servicePriceOf(service) {
  return Number(service?.price ?? service?.precio ?? 0);
}

function serviceAllowsVariablePrice(service) {
  return Boolean(
    service?.variablePrice ??
      service?.precioVariable ??
      service?.isVariablePrice ??
      service?.allowPriceOverride
  );
}

function servicePriceLabel(service) {
  const price = formatMoney(servicePriceOf(service));
  return serviceAllowsVariablePrice(service) ? `Desde ${price}` : price;
}

function itemSubtotal(item) {
  return Number(item.quantity || 1) * Number(item.unitPrice || 0);
}

function EmptyCard({ title, text, action }) {
  return <PremiumEmptyState title={title} message={text} action={action} />;
}
function CashNegativeAlert({ expected, cashSalesTotal, expense, labels = readBusinessLabels() }) {
    if (Number(expected || 0) >= 0) return null;
  
    return (
      <div className="rounded-[28px] border border-red-200 bg-red-50 p-5 shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="text-xs font-black uppercase tracking-[0.22em] text-red-600">
              Alerta de caja negativa
            </div>
  
            <h3 className="mt-2 text-xl font-black text-red-800">
              La caja física esperada está en negativo
            </h3>
  
            <p className="mt-2 max-w-3xl text-sm leading-6 text-red-700">
              Esto pasa cuando las salidas registradas superan el efectivo disponible.
              Revisa si hubo pagos a {labels.professionalsPlural}, adelantos o gastos registrados con un
              método incorrecto.
            </p>
          </div>
  
          <div className="rounded-2xl bg-white px-5 py-4 text-sm font-black text-red-700 shadow-sm">
            {formatMoney(cashSalesTotal)} - {formatMoney(expense)} = {formatMoney(expected)}
          </div>
        </div>
      </div>
  );
}

function PendingCashApprovalsSection({
  items,
  processingId,
  onApprove,
  onReject,
}) {
  if (!items.length) return null;

  return (
    <section className="rounded-[34px] border border-amber-200 bg-[linear-gradient(135deg,#FFFBEB_0%,#FFFFFF_70%)] p-6 shadow-[0_16px_45px_rgba(251,191,36,0.10)]">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-700">
            Aprobacion del dueno
          </div>
          <h3 className="mt-2 text-2xl font-black text-neutral-950">
            Movimientos pendientes de caja
          </h3>
          <p className="mt-1 max-w-3xl text-sm leading-6 text-neutral-600">
            Revisa ingresos u otros movimientos antes de que afecten el saldo esperado de caja.
          </p>
        </div>

        <div className="rounded-2xl bg-amber-100 px-4 py-3 text-sm font-black text-amber-800">
          {items.length} pendiente{items.length === 1 ? '' : 's'}
        </div>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-2">
        {items.map((movement) => {
          const busy = processingId === movement.id;

          return (
            <article
              key={movement.id}
              className="rounded-[26px] border border-amber-200 bg-white p-5 shadow-[0_12px_30px_rgba(15,23,42,0.05)]"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="text-xs font-black uppercase tracking-[0.16em] text-amber-700">
                    {movementTypeLabel(movement.type)} solicitado
                  </div>
                  <h4 className="mt-2 truncate text-lg font-black text-neutral-950">
                    {movement.concept || 'Movimiento pendiente'}
                  </h4>
                  <p className="mt-1 text-sm font-semibold text-neutral-500">
                    Solicitado por {movementRequestedBy(movement)}
                  </p>
                </div>

                <div className="text-right">
                  <div className={`text-2xl font-black ${movementAmountClass(movement.type)}`}>
                    {formatMoney(movement.amount)}
                  </div>
                  <div className="mt-1 text-xs font-bold text-neutral-400">
                    {methodLabel(movement.paymentMethod)}
                  </div>
                </div>
              </div>

              {movement.note && (
                <div className="mt-4 rounded-2xl border border-neutral-100 bg-neutral-50 px-4 py-3 text-sm font-semibold text-neutral-600">
                  {movement.note}
                </div>
              )}

              <div className="mt-5 flex flex-col gap-2 sm:flex-row sm:justify-end">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onReject(movement)}
                  className="rounded-2xl border border-red-200 bg-red-50 px-5 py-3 text-sm font-black text-red-700 transition hover:bg-red-100 disabled:opacity-60"
                >
                  Rechazar
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onApprove(movement)}
                  className="rounded-2xl bg-neutral-950 px-5 py-3 text-sm font-black text-white transition hover:scale-[1.01] disabled:opacity-60"
                >
                  {busy ? 'Procesando...' : 'Aprobar movimiento'}
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function PendingSaleValidationsSection({
  items,
  processingId,
  labels = readBusinessLabels(),
  onView,
  onApprove,
  onReject,
}) {
  if (!items.length) return null;

  return (
    <section className="rounded-[34px] border border-blue-200 bg-[linear-gradient(135deg,#EFF6FF_0%,#FFFFFF_72%)] p-6 shadow-[0_16px_45px_rgba(37,99,235,0.08)]">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="text-xs font-black uppercase tracking-[0.22em] text-blue-700">
            Ventas por validar
          </div>
          <h3 className="mt-2 text-2xl font-black text-neutral-950">
            Aprobar ventas del equipo
          </h3>
          <p className="mt-1 max-w-3xl text-sm leading-6 text-neutral-600">
            Revisa ventas registradas por {labels.professionalsPlural} antes de confirmarlas en caja y puntos.
          </p>
        </div>

        <div className="rounded-2xl bg-blue-100 px-4 py-3 text-sm font-black text-blue-800">
          {items.length} venta{items.length === 1 ? '' : 's'} pendiente{items.length === 1 ? '' : 's'}
        </div>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-2">
        {items.map((sale) => {
          const saleId = saleIdOf(sale);
          const busy = processingId === `sale-${saleId}`;
          const payments = salePaymentSummary(sale);

          return (
            <article
              key={saleId || `${saleDateOf(sale)}-${sale.total}`}
              className="rounded-[26px] border border-blue-200 bg-white p-5 shadow-[0_12px_30px_rgba(15,23,42,0.05)]"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="text-xs font-black uppercase tracking-[0.16em] text-blue-700">
                    Venta #{saleId || '-'}
                  </div>
                  <h4 className="mt-2 truncate text-lg font-black text-neutral-950">
                    {sale.customerName || 'Cliente ocasional'}
                  </h4>
                  <p className="mt-1 text-sm font-semibold text-neutral-500">
                    Registrada por {saleCreatedBy(sale)}
                  </p>
                </div>

                <div className="text-right">
                  <div className="text-2xl font-black text-emerald-700">
                    {formatMoney(sale.total)}
                  </div>
                  <div className="mt-1 text-xs font-bold text-neutral-400">
                    {methodLabel(sale.metodoPago)}
                  </div>
                </div>
              </div>

              <div className="mt-4 grid gap-2 sm:grid-cols-2">
                <div className="rounded-2xl border border-neutral-100 bg-neutral-50 px-4 py-3">
                  <div className="text-xs font-black uppercase tracking-[0.14em] text-neutral-400">
                    {labels.professionalSingular}
                  </div>
                  <div className="mt-1 truncate text-sm font-black text-neutral-800">
                    {saleBarberName(sale)}
                  </div>
                </div>
                <div className="rounded-2xl border border-neutral-100 bg-neutral-50 px-4 py-3">
                  <div className="text-xs font-black uppercase tracking-[0.14em] text-neutral-400">
                    Fecha
                  </div>
                  <div className="mt-1 truncate text-sm font-black text-neutral-800">
                    {formatDateTime(saleDateOf(sale))}
                  </div>
                </div>
              </div>

              <div className="mt-4 rounded-2xl border border-neutral-100 bg-neutral-50 px-4 py-3 text-sm font-semibold text-neutral-600">
                {payments}
              </div>

              <div className="mt-5 flex flex-col gap-2 sm:flex-row sm:justify-end">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onView(sale)}
                  className="rounded-2xl border border-neutral-200 bg-white px-5 py-3 text-sm font-black text-neutral-700 transition hover:bg-neutral-50 disabled:opacity-60"
                >
                  Ver detalle
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onReject(sale)}
                  className="rounded-2xl border border-red-200 bg-red-50 px-5 py-3 text-sm font-black text-red-700 transition hover:bg-red-100 disabled:opacity-60"
                >
                  Rechazar
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onApprove(sale)}
                  className="rounded-2xl bg-neutral-950 px-5 py-3 text-sm font-black text-white transition hover:scale-[1.01] disabled:opacity-60"
                >
                  {busy ? 'Procesando...' : 'Aprobar venta'}
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function OpenCashModal({ branch, canManageFund = false, onClose, onSaved }) {
  const [openingAmount, setOpeningAmount] = useState('0');
  const [fundWithdrawalAmount, setFundWithdrawalAmount] = useState('0');
  const [openingNote, setOpeningNote] = useState('');
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMsg('');

    const amount = Number(String(openingAmount).replace(',', '.'));
    const fundWithdrawal = canManageFund
      ? Number(String(fundWithdrawalAmount).replace(',', '.')) || 0
      : 0;

    if (Number.isNaN(amount) || amount < 0) {
      setErrorMsg('Ingresa un monto válido.');
      return;
    }

    if (Number.isNaN(fundWithdrawal) || fundWithdrawal < 0) {
      setErrorMsg('Ingresa un monto valido para tomar del fondo.');
      return;
    }

    setSaving(true);

    try {
      await openCashRegister({
        branchId: branch.id,
        openingAmount: amount,
        fundWithdrawalAmount: fundWithdrawal,
        openingNote: openingNote.trim() || null,
      });

      onSaved();
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo abrir la caja.');
    } finally {
      setSaving(false);
    }
  }

  return (
    <ModalShell title="Abrir caja" subtitle={branch?.name || 'Sede'} onClose={onClose}>
      <form onSubmit={handleSubmit} className="space-y-4">
        <InputField
          label="Monto inicial"
          value={openingAmount}
          onChange={setOpeningAmount}
          type="number"
          step="0.01"
          prefix={getTenantCurrencySymbol()}
        />

        {canManageFund && (
          <>
        <InputField
          label="Tomar del fondo acumulado"
          value={fundWithdrawalAmount}
          onChange={setFundWithdrawalAmount}
          type="number"
          step="0.01"
          prefix={getTenantCurrencySymbol()}
        />

        <p className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs font-bold leading-5 text-amber-800">
          Opcional: pasa dinero guardado al efectivo inicial de esta caja.
        </p>
          </>
        )}

        <TextAreaField
          label="Nota de apertura"
          value={openingNote}
          onChange={setOpeningNote}
          placeholder="Ej. Apertura de turno mañana"
        />

        {errorMsg && <ErrorBox message={errorMsg} />}

        <button
          disabled={saving}
          className="w-full rounded-2xl bg-amber-400 px-5 py-4 font-black text-neutral-950 transition hover:scale-[1.01] disabled:opacity-60"
        >
          {saving ? 'Abriendo...' : 'Abrir caja'}
        </button>
      </form>
    </ModalShell>
  );
}

function CloseCashModal({ branch, cashRegister, canManageFund = false, onClose, onSaved }) {
  const expected = Number(cashRegister?.closingAmountExpected || 0);

  const [counted, setCounted] = useState(String(expected.toFixed(2)));
  const [fundDepositAmount, setFundDepositAmount] = useState('0');
  const [closingNote, setClosingNote] = useState('');
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const countedNumber = Number(String(counted).replace(',', '.')) || 0;
  const fundDepositNumber = canManageFund
    ? Number(String(fundDepositAmount).replace(',', '.')) || 0
    : 0;
  const difference = countedNumber - expected;

  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMsg('');

    if (Number.isNaN(countedNumber) || countedNumber < 0) {
      setErrorMsg('Ingresa un monto contado válido.');
      return;
    }

    if (Number.isNaN(fundDepositNumber) || fundDepositNumber < 0) {
      setErrorMsg('Ingresa un monto valido para enviar al fondo.');
      return;
    }

    if (fundDepositNumber > countedNumber) {
      setErrorMsg('No puedes enviar al fondo mas de lo contado.');
      return;
    }

    setSaving(true);

    try {
      await closeCashRegister({
        branchId: branch.id,
        cashRegisterId: cashRegister.id,
        closingAmountCounted: countedNumber,
        fundDepositAmount: fundDepositNumber,
        closingNote: closingNote.trim() || null,
      });

      onSaved();
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo cerrar la caja.');
    } finally {
      setSaving(false);
    }
  }

  return (
    <ModalShell title="Cerrar caja" subtitle={branch?.name || 'Sede'} onClose={onClose}>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <StatCard title="Esperado" value={formatMoney(expected)} helper="Según sistema" tone="gold" />
          <StatCard
            title="Diferencia"
            value={formatMoney(difference)}
            helper={difference === 0 ? 'Cuadra perfecto' : difference > 0 ? 'Sobra dinero' : 'Falta dinero'}
            tone={difference === 0 ? 'green' : 'red'}
          />
        </div>

        <InputField
          label="Monto contado"
          value={counted}
          onChange={setCounted}
          type="number"
          step="0.01"
          prefix={getTenantCurrencySymbol()}
        />

        {canManageFund && (
          <>
        <InputField
          label="Enviar al fondo acumulado"
          value={fundDepositAmount}
          onChange={setFundDepositAmount}
          type="number"
          step="0.01"
          prefix={getTenantCurrencySymbol()}
        />

        <p className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs font-bold leading-5 text-amber-800">
          Este monto queda como colchon acumulado para gastos grandes futuros.
        </p>
          </>
        )}

        <TextAreaField
          label="Observación de cierre"
          value={closingNote}
          onChange={setClosingNote}
          placeholder="Ej. Caja cerrada sin diferencias"
        />

        {errorMsg && <ErrorBox message={errorMsg} />}

        <button
          disabled={saving}
          className="w-full rounded-2xl bg-neutral-950 px-5 py-4 font-black text-white transition hover:scale-[1.01] disabled:opacity-60"
        >
          {saving ? 'Cerrando...' : 'Cerrar caja'}
        </button>
      </form>
    </ModalShell>
  );
}

function ReconciliationModal({ branch, cashRegister, canManageFund = false, onClose, onSaved }) {
  const expected = Number(cashRegister?.closingAmountExpected || 0);
  const [counted, setCounted] = useState(String(expected.toFixed(2)));
  const [fundDeposits, setFundDeposits] = useState({});
  const [note, setNote] = useState('');
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const balances = Array.isArray(cashRegister?.paymentMethodBalances)
    ? cashRegister.paymentMethodBalances
    : [];

  async function submit(event) {
    event.preventDefault();
    const countedAmount = Number(String(counted).replace(',', '.'));
    if (!Number.isFinite(countedAmount) || countedAmount < 0) {
      setErrorMsg('Ingresa el efectivo contado correctamente.');
      return;
    }
    const deposits = {};
    if (canManageFund) {
      for (const [method, raw] of Object.entries(fundDeposits)) {
        const amount = Number(String(raw || 0).replace(',', '.'));
        if (amount > 0) deposits[method] = amount;
      }
    }
    setSaving(true);
    setErrorMsg('');
    try {
      await reconcileCashRegister({
        branchId: branch.id,
        cashRegisterId: cashRegister.id,
        closingAmountCounted: countedAmount,
        fundDeposits: deposits,
        note: note.trim() || null,
      });
      onSaved();
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo conciliar la caja.');
    } finally {
      setSaving(false);
    }
  }

  return (
    <ModalShell title="Conciliar cierre automatico" subtitle={branch?.name || 'Sede'} onClose={onClose}>
      <form onSubmit={submit} className="space-y-4">
        <StatCard title="Esperado por sistema" value={formatMoney(expected)} helper="Revisa gastos olvidados antes de confirmar" tone="gold" />
        <InputField label="Efectivo contado" value={counted} onChange={setCounted} type="number" step="0.01" prefix={getTenantCurrencySymbol()} />
        {canManageFund && (
        <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
          <div className="font-black text-neutral-950">Enviar al fondo acumulado</div>
          <div className="mt-1 text-xs font-semibold text-neutral-500">Elige cuanto guardar por cada metodo. Puedes dejar todo en cero.</div>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {balances.map((row) => {
              const method = normalizeMethod(row.paymentMethod);
              return (
                <InputField
                  key={method}
                  label={`${methodLabel(method)} · disponible ${formatMoney(row.totalAmount)}`}
                  value={fundDeposits[method] || ''}
                  onChange={(value) => setFundDeposits((current) => ({ ...current, [method]: value }))}
                  type="number"
                  step="0.01"
                  prefix={getTenantCurrencySymbol()}
                />
              );
            })}
          </div>
        </div>
        )}
        <TextAreaField label="Observacion" value={note} onChange={setNote} placeholder="Ej. Efectivo confirmado y saldo enviado al fondo." />
        {errorMsg && <ErrorBox message={errorMsg} />}
        <button disabled={saving} className="w-full rounded-2xl bg-amber-400 px-5 py-4 font-black text-neutral-950 disabled:opacity-60">
          {saving ? 'Conciliando...' : 'Confirmar conciliacion'}
        </button>
      </form>
    </ModalShell>
  );
}
function MovementModal({ branch, cashRegister, paymentMethods = DEFAULT_PAYMENT_METHODS, initialMovement = null, canManageFund = false, labels = readBusinessLabels(), onClose, onSaved }) {
  const isEditing = Boolean(initialMovement?.id);
  const [type, setType] = useState(initialMovement?.type || 'EXPENSE');
  const [amount, setAmount] = useState(initialMovement?.amount ? String(initialMovement.amount) : '');
  const [concept, setConcept] = useState(initialMovement?.concept || 'Gasto general');
  const [note, setNote] = useState(initialMovement?.note || '');
  const [movementDate, setMovementDate] = useState(
    initialMovement?.movementDate
      ? String(initialMovement.movementDate).slice(0, 10)
      : toDateInputValue(new Date())
  );
  const [auditReason, setAuditReason] = useState('');
  const [paymentMethod, setPaymentMethod] = useState(normalizeMethod(initialMovement?.paymentMethod) || 'CASH');
  const [fundingSource, setFundingSource] = useState(
    canManageFund ? initialMovement?.fundingSource || 'CASH_REGISTER' : 'CASH_REGISTER'
  );
  const [fromPaymentMethod, setFromPaymentMethod] = useState(normalizeMethod(initialMovement?.fromPaymentMethod) || defaultExtraPaymentMethod(paymentMethods));
  const [toPaymentMethod, setToPaymentMethod] = useState(normalizeMethod(initialMovement?.toPaymentMethod) || 'CASH');

  const [barbers, setBarbers] = useState([]);
  const [selectedBarberId, setSelectedBarberId] = useState(String(initialMovement?.barberUserId ?? initialMovement?.barberId ?? ''));
  const [loadingBarbers, setLoadingBarbers] = useState(false);

  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const concepts = {
    INCOME: [
      'Ingreso adicional',
      'Abono manual',
      'Regularización de caja',
      'Pago externo',
      'Otro ingreso',
    ],
    EXPENSE: [
      'Regularización de caja',
      'Compra de productos',
      'Limpieza',
      'Movilidad',
      'Alimentación',
      'Mantenimiento',
      'Pago de Luz',
      'Pago de Agua',
      'Pago de Alquiler',
      'Servicios',
      'Otros',
    ],
    ADVANCE_BARBER: [
      'Adelanto a ' + labels.professionalSingular,
      'Préstamo del día',
      'Liquidación parcial',
    ],
    PAYMENT_BARBER: [
      'Pago a ' + labels.professionalSingular + ' manual',
      'Pago de porcentaje',
      'Pago de comisión',
    ],
    PAYMENT_METHOD_TRANSFER: [
      'Traslado entre métodos',
      'Conversión a efectivo',
      'Depósito a cuenta digital',
      'Regularización entre métodos',
    ],
  };

  useEffect(() => {
    let alive = true;
    setLoadingBarbers(true);
    getCashBarbers(branch?.id)
      .then((data) => { if (alive) setBarbers(Array.isArray(data) ? data : []); })
      .catch(() => { if (alive) setBarbers([]); })
      .finally(() => { if (alive) setLoadingBarbers(false); });
    return () => { alive = false; };
  }, [branch?.id]);

  const needsBarber = type === 'ADVANCE_BARBER' || type === 'PAYMENT_BARBER';
  const isOutflow = type === 'EXPENSE' || type === 'ADVANCE_BARBER' || type === 'PAYMENT_BARBER';
  const isTransfer = type === 'PAYMENT_METHOD_TRANSFER';
  const sourceOptions = fundingSourceSelectOptions(cashRegister, paymentMethod, canManageFund);
  const paymentOptions = isOutflow
    ? paymentSelectOptionsWithBalances(paymentMethods, cashRegister, fundingSource)
    : paymentSelectOptions(paymentMethods);

  useEffect(() => {
    async function loadBarbers() {
      if (!branch?.id) return;

      setLoadingBarbers(true);

      try {
        const [professionalsResult, internalUsersResult] = await Promise.allSettled([
          getCashBarbers(branch.id),
          getCashPayableEmployees(branch.id),
        ]);
        if (professionalsResult.status === 'rejected') {
          throw professionalsResult.reason;
        }
        const professionals = professionalsResult.value || [];
        const internalUsers = internalUsersResult.status === 'fulfilled'
          ? internalUsersResult.value || []
          : [];
        const payableEmployees = internalUsers
          .filter((item) => item.activo && item.salaryEnabled)
          .filter((item) => item.branchIds.includes(Number(branch.id)) || Number(item.branchId) === Number(branch.id))
          .map((item) => ({ ...item, name: item.fullName, role: item.rol, employeeOnly: !item.professionalProfileEnabled }));
        setBarbers([...professionals, ...payableEmployees]
          .filter((item) => item.id > 0)
          .filter((item, index, list) => list.findIndex((candidate) => Number(candidate.id) === Number(item.id)) === index));
      } catch {
        setBarbers([]);
      } finally {
        setLoadingBarbers(false);
      }
    }

    loadBarbers();
  }, [branch?.id]);

  function handleTypeChange(nextType) {
    setType(nextType);
    setConcept(concepts[nextType]?.[0] || 'Movimiento');

    if (nextType !== 'ADVANCE_BARBER' && nextType !== 'PAYMENT_BARBER') {
      setSelectedBarberId('');
    }

    if (nextType === 'PAYMENT_METHOD_TRANSFER') {
      setFromPaymentMethod(defaultExtraPaymentMethod(paymentMethods));
      setToPaymentMethod('CASH');
    }
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMsg('');

    const parsedAmount = Number(String(amount).replace(',', '.'));

    if (Number.isNaN(parsedAmount) || parsedAmount <= 0) {
      setErrorMsg('Ingresa un monto mayor a cero.');
      return;
    }

    if (!isEditing && isOutflow && fundingSource !== 'EXTERNAL') {
      const available = availableBalanceForMethod(cashRegister, fundingSource, paymentMethod);
      if (parsedAmount > available + 0.009) {
        setErrorMsg(
          `No hay saldo suficiente en ${methodLabel(paymentMethod)}. Disponible: ${formatMoney(available)}.`
        );
        return;
      }
    }

    if (isTransfer && fromPaymentMethod === toPaymentMethod) {
      setErrorMsg('El método origen y destino no pueden ser iguales.');
      return;
    }

    if (needsBarber && !selectedBarberId) {
      setErrorMsg('Selecciona el ' + labels.professionalSingular + ' relacionado.');
      return;
    }

    if (isEditing && !auditReason.trim()) {
      setErrorMsg('Escribe el motivo de auditoria para editar este movimiento.');
      return;
    }

    setSaving(true);

    try {
      const payload = {
        branchId: branch.id,
        type,
        amount: parsedAmount,
        concept,
        note: note.trim() || null,
        barberUserId: needsBarber ? Number(selectedBarberId) : null,
        paymentMethod: isTransfer ? toPaymentMethod : paymentMethod,
        fundingSource: isOutflow ? fundingSource : 'CASH_REGISTER',
        fromPaymentMethod: isTransfer ? fromPaymentMethod : null,
        toPaymentMethod: isTransfer ? toPaymentMethod : null,
        movementDate,
        auditReason: isEditing ? auditReason.trim() : null,
      };

      if (isEditing) {
        await updateCashMovement({
          ...payload,
          movementId: initialMovement.id,
        });
      } else {
        await createCashMovement({
          ...payload,
          cashRegisterId: cashRegister.id,
        });
      }

      onSaved();
    } catch (error) {
      setErrorMsg(error.message || (isEditing ? 'No se pudo actualizar el movimiento.' : 'No se pudo registrar el movimiento.'));
    } finally {
      setSaving(false);
    }
  }

  return (
    <ModalShell title={isEditing ? 'Editar movimiento' : 'Registrar movimiento'} subtitle={branch?.name || 'Sede'} onClose={onClose}>
      <form onSubmit={handleSubmit} className="space-y-4">
        {!isEditing && (
          <div className="rounded-2xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm font-semibold leading-6 text-blue-800">
            Para cuadrar hoy: registra un ingreso si tienes más dinero físico que el sistema, o una salida si tienes menos. Usa “Regularización de caja” y deja el motivo en la nota.
          </div>
        )}

        <SelectField
          label="Tipo de movimiento"
          value={type}
          onChange={handleTypeChange}
          options={[
            { value: 'INCOME', label: 'Ingresar dinero / ajuste a favor' },
            { value: 'EXPENSE', label: 'Sacar dinero / registrar gasto' },
            { value: 'ADVANCE_BARBER', label: 'Adelanto a ' + labels.professionalSingular },
            { value: 'PAYMENT_BARBER', label: 'Pago a ' + labels.professionalSingular + ' manual' },
            { value: 'PAYMENT_METHOD_TRANSFER', label: 'Trasladar dinero entre métodos' },
          ]}
        />

        <InputField
          label="Monto"
          value={amount}
          onChange={setAmount}
          type="number"
          step="0.01"
          prefix={getTenantCurrencySymbol()}
        />

        <InputField
          label="Fecha del movimiento"
          value={movementDate}
          onChange={setMovementDate}
          type="date"
        />

        <SelectField
          label="Concepto"
          value={concept}
          onChange={setConcept}
          options={(concepts[type] || ['Movimiento']).map((item) => ({
            value: item,
            label: item,
          }))}
        />

        {needsBarber && (
          <>
            {loadingBarbers ? (
              <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 text-sm font-black text-neutral-500">
                Cargando {labels.professionalsPlural}...
              </div>
            ) : barbers.length === 0 ? (
              <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-4 text-sm font-bold text-red-700">
                No se encontraron {labels.professionalsPlural} activos para esta sede.
              </div>
            ) : (
              <SelectField
                label={labels.professionalDisplay + " relacionado"}
                value={selectedBarberId}
                onChange={setSelectedBarberId}
                options={[
                  { value: '', label: 'Selecciona un ' + labels.professionalSingular },
                  ...barbers.map((barber) => ({
                    value: String(barber.id),
                    label: barber.name,
                  })),
                ]}
              />
            )}
          </>
        )}

        {isOutflow && (
          <SelectField
            label="Sale desde"
            value={fundingSource}
            onChange={setFundingSource}
            disabled={isEditing}
            options={sourceOptions}

          />
        )}
        {isTransfer ? (
          <div className="grid gap-3 sm:grid-cols-2">
            <SelectField
              label="Desde"
              value={fromPaymentMethod}
              onChange={setFromPaymentMethod}
              options={paymentOptions}
            />

            <SelectField
              label="Hacia"
              value={toPaymentMethod}
              onChange={setToPaymentMethod}
              options={paymentOptions}
            />
          </div>
        ) : (
          <SelectField
            label={type === 'INCOME' ? 'Método de ingreso' : 'Método de salida'}
            value={paymentMethod}
            onChange={setPaymentMethod}
            options={paymentOptions}
          />
        )}

        <div className="rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-semibold text-emerald-800">
          {isTransfer
            ? 'Este traslado no crea una venta. Solo mueve saldo de un método a otro.'
            : type === 'INCOME'
              ? 'Este ingreso se sumará al esperado de caja solo si el método es efectivo.'
              : needsBarber
                ? 'Este movimiento quedará relacionado al ' + labels.professionalSingular + ' seleccionado.'
                : 'Solo los movimientos en efectivo afectarán el esperado de caja.'}
        </div>

        <TextAreaField
          label="Nota"
          value={note}
          onChange={setNote}
          placeholder={needsBarber ? 'Ej. Pago manual o adelanto solicitado.' : 'Detalle opcional'}
        />

        {isEditing && (
          <TextAreaField
            label="Motivo de auditoria"
            value={auditReason}
            onChange={setAuditReason}
            placeholder="Ej. Correccion de monto, profesional o metodo de pago."
          />
        )}

        {errorMsg && <ErrorBox message={errorMsg} />}

        <button
          disabled={saving}
          className="w-full rounded-2xl bg-amber-400 px-5 py-4 font-black text-neutral-950 transition hover:scale-[1.01] disabled:opacity-60"
        >
          {saving ? 'Guardando...' : isEditing ? 'Guardar cambios' : 'Guardar movimiento'}
        </button>
      </form>
    </ModalShell>
  );
}

function FundMovementModal({ branch, paymentMethods = DEFAULT_PAYMENT_METHODS, onClose, onSaved }) {
  const [type, setType] = useState('MANUAL_DEPOSIT');
  const [paymentMethod, setPaymentMethod] = useState('CASH');
  const [amount, setAmount] = useState('');
  const [concept, setConcept] = useState('Ingreso manual al fondo');
  const [note, setNote] = useState('');
  const [movementDate, setMovementDate] = useState(toDateInputValue(new Date()));
  const [summary, setSummary] = useState({ totalBalance: 0, balances: [] });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  useEffect(() => {
    let active = true;
    setLoading(true);
    getCashFundSummary(branch.id)
      .then((data) => {
        if (active) {
          setSummary({
            totalBalance: Number(data?.totalBalance || 0),
            balances: Array.isArray(data?.balances) ? data.balances : [],
          });
        }
      })
      .catch((error) => {
        if (active) setErrorMsg(error.message || 'No se pudo cargar el fondo acumulado.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => { active = false; };
  }, [branch.id]);

  const fundCashRegister = {
    accumulatedFundBalance: summary.totalBalance,
    accumulatedFundBalances: summary.balances,
  };
  const available = availableBalanceForMethod(
    fundCashRegister,
    'ACCUMULATED_FUND',
    paymentMethod
  );
  const paymentOptions = paymentSelectOptions(paymentMethods).map((option) => ({
    ...option,
    label: `${option.label} · disponible ${formatMoney(
      availableBalanceForMethod(fundCashRegister, 'ACCUMULATED_FUND', option.value)
    )}`,
  }));
  const isWithdrawal = type === 'MANUAL_WITHDRAWAL';

  function handleTypeChange(value) {
    setType(value);
    setConcept(
      value === 'MANUAL_WITHDRAWAL'
        ? 'Retiro manual del fondo'
        : 'Ingreso manual al fondo'
    );
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setErrorMsg('');
    const parsedAmount = roundMoney(parseMoneyInput(amount));

    if (parsedAmount <= 0) {
      setErrorMsg('Ingresa un monto mayor a cero.');
      return;
    }
    if (isWithdrawal && parsedAmount > available + 0.009) {
      setErrorMsg(
        `No hay saldo suficiente en ${methodLabel(paymentMethod)}. Disponible: ${formatMoney(available)}.`
      );
      return;
    }
    if (!concept.trim()) {
      setErrorMsg('Escribe el motivo del movimiento para dejar auditoría.');
      return;
    }

    setSaving(true);
    try {
      await createCashFundMovement({
        branchId: branch.id,
        type,
        paymentMethod,
        amount: parsedAmount,
        concept: concept.trim(),
        note: note.trim() || null,
        movementDate,
      });
      await onSaved();
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo actualizar el fondo acumulado.');
    } finally {
      setSaving(false);
    }
  }

  return (
    <ModalShell title="Gestionar fondo acumulado" subtitle={branch?.name || 'Sede'} onClose={onClose}>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <MiniPreviewItem label="Saldo total del fondo" value={formatMoney(summary.totalBalance)} strong />
          <MiniPreviewItem
            label={`${methodLabel(paymentMethod)} disponible`}
            value={loading ? 'Cargando...' : formatMoney(available)}
          />
        </div>

        <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold leading-6 text-amber-800">
          Este formulario modifica el fondo guardado, no el saldo de la caja abierta. Para cuadrar el dinero físico de hoy usa “Ingresar / sacar dinero”.
        </div>

        <SelectField
          label="Operación"
          value={type}
          onChange={handleTypeChange}
          options={[
            { value: 'MANUAL_DEPOSIT', label: 'Ingresar dinero al fondo' },
            { value: 'MANUAL_WITHDRAWAL', label: 'Retirar dinero del fondo' },
          ]}
        />

        <SelectField
          label="Método donde está el dinero"
          value={paymentMethod}
          onChange={setPaymentMethod}
          options={paymentOptions}
        />

        <div className="grid gap-3 sm:grid-cols-2">
          <InputField
            label="Monto"
            value={amount}
            onChange={setAmount}
            type="number"
            step="0.01"
            prefix={getTenantCurrencySymbol()}
          />
          <InputField
            label="Fecha"
            value={movementDate}
            onChange={setMovementDate}
            type="date"
          />
        </div>

        <InputField label="Motivo" value={concept} onChange={setConcept} />
        <TextAreaField
          label="Nota"
          value={note}
          onChange={setNote}
          placeholder="Ej. Dinero reservado por el dueño o retiro para depósito bancario."
        />

        {errorMsg && <ErrorBox message={errorMsg} />}

        <button
          disabled={saving || loading}
          className="w-full rounded-2xl bg-amber-400 px-5 py-4 font-black text-neutral-950 disabled:opacity-60"
        >
          {saving ? 'Guardando...' : isWithdrawal ? 'Confirmar retiro' : 'Confirmar ingreso'}
        </button>
      </form>
    </ModalShell>
  );
}

function BarberPaymentModal({ branch, cashRegister, paymentMethods = DEFAULT_PAYMENT_METHODS, canManageFund = false, labels = readBusinessLabels(), onClose, onSaved }) {
  const today = new Date();
  const sevenDaysAgo = new Date();
  sevenDaysAgo.setDate(today.getDate() - 6);

  const [barbers, setBarbers] = useState([]);
  const [selectedBarberId, setSelectedBarberId] = useState('');
  const [periodFrom, setPeriodFrom] = useState(toDateInputValue(sevenDaysAgo));
  const [periodTo, setPeriodTo] = useState(toDateInputValue(today));
  const [movementDate, setMovementDate] = useState(toDateInputValue(today));
  const [amountPaid, setAmountPaid] = useState('');
  const [paymentRows, setPaymentRows] = useState([createPaymentDraft('CASH', 0)]);
  const [fundingSource, setFundingSource] = useState('CASH_REGISTER');
  const [note, setNote] = useState('');

  const [preview, setPreview] = useState(null);
  const [loadingBarbers, setLoadingBarbers] = useState(true);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const sourceOptions = fundingSourceSelectOptions(cashRegister, null, canManageFund);
  const paymentOptions = paymentSelectOptionsWithBalances(
    paymentMethods,
    cashRegister,
    fundingSource
  );

  const amountToPay = roundMoney(parseMoneyInput(amountPaid));
  const distributedTotal = useMemo(
    () => roundMoney(paymentRows.reduce((sum, row) => sum + parseMoneyInput(row.amount), 0)),
    [paymentRows]
  );
  const remainingToDistribute = roundMoney(amountToPay - distributedTotal);

  useEffect(() => {
    async function loadBarbers() {
      setLoadingBarbers(true);
      setErrorMsg('');

      try {
        const [professionalsResult, internalUsersResult] = await Promise.allSettled([
          getCashBarbers(branch.id),
          getCashPayableEmployees(branch.id),
        ]);
        if (professionalsResult.status === 'rejected') {
          throw professionalsResult.reason;
        }
        const professionals = professionalsResult.value || [];
        const internalUsers = internalUsersResult.status === 'fulfilled'
          ? internalUsersResult.value || []
          : [];
        const payableEmployees = internalUsers
          .filter((item) => item.activo && item.salaryEnabled)
          .filter((item) => item.branchIds.includes(Number(branch.id)) || Number(item.branchId) === Number(branch.id))
          .map((item) => ({ ...item, name: item.fullName, role: item.rol, employeeOnly: !item.professionalProfileEnabled }));
        setBarbers([...professionals, ...payableEmployees]
          .filter((item) => item.id > 0)
          .filter((item, index, list) => list.findIndex((candidate) => Number(candidate.id) === Number(item.id)) === index));
      } catch (error) {
        setErrorMsg(error.message || 'No se pudieron cargar los ' + labels.professionalsPlural + '.');
      } finally {
        setLoadingBarbers(false);
      }
    }

    loadBarbers();
  }, [branch.id]);

  function resetPaymentRows(total, preferredMethod = 'CASH') {
    const methodExists = paymentOptions.some((option) => option.value === preferredMethod);
    const method = methodExists ? preferredMethod : paymentOptions[0]?.value || 'CASH';
    setPaymentRows([createPaymentDraft(method, Math.max(Number(total || 0), 0))]);
  }

  async function loadPreview(nextBarberId = selectedBarberId) {
    if (!nextBarberId) return;

    setLoadingPreview(true);
    setErrorMsg('');

    try {
      const data = await getBarberPaymentPreview({
        branchId: branch.id,
        barberUserId: Number(nextBarberId),
        periodFrom,
        periodTo,
        movementDate,
      });

      setPreview(data);

      const pending = Number(data?.pendingAmount ?? 0);
      const nextAmount = pending > 0 ? pending.toFixed(2) : '0.00';
      setAmountPaid(nextAmount);
      resetPaymentRows(pending > 0 ? pending : 0);
    } catch (error) {
      setPreview(null);
      setErrorMsg(error.message || 'No se pudo calcular el pago del ' + labels.professionalSingular + '.');
    } finally {
      setLoadingPreview(false);
    }
  }

  async function handleBarberChange(value) {
    setSelectedBarberId(value);
    setPreview(null);
    setAmountPaid('');
    resetPaymentRows(0);

    if (value) {
      await loadPreview(value);
    }
  }

  function handleDateChange(kind, value) {
    if (kind === 'from') {
      setPeriodFrom(value);
    } else {
      setPeriodTo(value);
    }

    setPreview(null);
    setAmountPaid('');
    resetPaymentRows(0);
  }

  function handleAmountChange(value) {
    setAmountPaid(value);

    const total = parseMoneyInput(value);
    if (paymentRows.length === 1) {
      setPaymentRows((current) => [
        {
          ...current[0],
          amount: total > 0 ? total.toFixed(2) : '0.00',
        },
      ]);
    }
  }

  function updatePaymentRow(key, field, value) {
    setPaymentRows((current) =>
      current.map((row) =>
        row.key === key
          ? {
              ...row,
              [field]: field === 'method' ? normalizeMethod(value) : value,
            }
          : row
      )
    );
  }

  function addPaymentRow() {
    const nextMethod = paymentOptions.find((option) => !paymentRows.some((row) => normalizeMethod(row.method) === option.value))?.value
      || paymentOptions[0]?.value
      || 'CASH';

    const amount = remainingToDistribute > 0 ? remainingToDistribute : 0;

    setPaymentRows((current) => [
      ...current,
      createPaymentDraft(nextMethod, amount),
    ]);
  }

  function removePaymentRow(key) {
    setPaymentRows((current) => {
      if (current.length <= 1) return current;
      return current.filter((row) => row.key !== key);
    });
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMsg('');

    if (!selectedBarberId) {
      setErrorMsg('Selecciona un ' + labels.professionalSingular + '.');
      return;
    }

    const amount = roundMoney(parseMoneyInput(amountPaid));

    if (amount <= 0) {
      setErrorMsg('Ingresa un monto válido mayor a cero.');
      return;
    }

    const normalizedPayments = paymentRows
      .map((row) => ({
        method: normalizeMethod(row.method),
        amount: roundMoney(parseMoneyInput(row.amount)),
      }))
      .filter((row) => row.method && row.amount > 0);

    if (normalizedPayments.length === 0) {
      setErrorMsg('Agrega al menos un método de pago.');
      return;
    }

    const totalPayments = roundMoney(
      normalizedPayments.reduce((sum, row) => sum + row.amount, 0)
    );

    if (Math.abs(totalPayments - amount) > 0.009) {
      setErrorMsg(
        `La distribución debe sumar exactamente ${formatMoney(amount)}. Actualmente suma ${formatMoney(totalPayments)}.`
      );
      return;
    }

    if (fundingSource !== 'EXTERNAL') {
      const requestedByMethod = normalizedPayments.reduce((totals, payment) => {
        totals[payment.method] = roundMoney((totals[payment.method] || 0) + payment.amount);
        return totals;
      }, {});

      for (const [method, requested] of Object.entries(requestedByMethod)) {
        const available = availableBalanceForMethod(cashRegister, fundingSource, method);
        if (requested > available + 0.009) {
          setErrorMsg(
            `No hay saldo suficiente en ${methodLabel(method)}. Disponible: ${formatMoney(available)}.`
          );
          return;
        }
      }
    }

    setSaving(true);

    try {
      await createBarberPayment({
        branchId: branch.id,
        cashRegisterId: cashRegister.id,
        barberUserId: Number(selectedBarberId),
        periodFrom,
        periodTo,
        movementDate,
        amountPaid: amount,
        paymentMethod: normalizedPayments.length > 1 ? 'MIXED' : normalizedPayments[0].method,
        fundingSource,
        payments: normalizedPayments,
        note: note.trim() || null,
      });

      onSaved();
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo registrar el pago al ' + labels.professionalSingular + '.');
    } finally {
      setSaving(false);
    }
  }

  return (
    <ModalShell title={'Pagar al equipo'} subtitle={branch?.name || 'Sede'} onClose={onClose}>
      <form onSubmit={handleSubmit} className="space-y-4">
        {loadingBarbers ? (
          <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 text-sm font-black text-neutral-500">
            Cargando {labels.professionalsPlural}...
          </div>
        ) : (
          <SelectField
            label={labels.professionalDisplay}
            value={selectedBarberId}
            onChange={handleBarberChange}
            options={[
              { value: '', label: 'Selecciona un ' + labels.professionalSingular },
              ...barbers.map((barber) => ({
                value: String(barber.id),
                label: barber.name,
              })),
            ]}
          />
        )}

        <div className="grid gap-3 sm:grid-cols-3">
          <InputField
            label="Desde"
            value={periodFrom}
            onChange={(value) => handleDateChange('from', value)}
            type="date"
          />

          <InputField
            label="Hasta"
            value={periodTo}
            onChange={(value) => handleDateChange('to', value)}
            type="date"
          />

          <InputField
            label="Fecha en caja"
            value={movementDate}
            onChange={setMovementDate}
            type="date"
          />
        </div>

        <SelectField
          label="Sale desde"
          value={fundingSource}
          onChange={setFundingSource}
          options={sourceOptions}
        />

        <div className="rounded-2xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm font-semibold text-blue-800">
          Los saldos se validan por método. Si eliges fondo acumulado, el pago descontará el fondo del método seleccionado; la cuenta externa no altera la caja.
        </div>

        <button
          type="button"
          onClick={() => loadPreview()}
          disabled={!selectedBarberId || loadingPreview}
          className="w-full rounded-2xl border border-neutral-200 bg-neutral-950 px-5 py-4 text-sm font-black text-white transition hover:scale-[1.01] disabled:opacity-60"
        >
          {loadingPreview ? 'Calculando...' : 'Calcular pendiente'}
        </button>

        {preview && (
          <div className="rounded-[26px] border border-amber-200 bg-amber-50 p-5">
            <div className="text-xs font-black uppercase tracking-[0.2em] text-amber-700">
              Resumen inteligente
            </div>

            <div className="mt-3 text-xl font-black text-neutral-950">
              {preview.barberName || labels.professionalDisplay}
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {String(preview.paymentMode || '').toUpperCase() === 'SALARY' ? (
                <MiniPreviewItem label="Sueldo del periodo" value={formatMoney(preview.salaryAmount)} />
              ) : (
                <MiniPreviewItem label="Servicios" value={formatMoney(preview.serviceCommissionAmount)} />
              )}
              <MiniPreviewItem label="Productos" value={formatMoney(preview.productCommissionAmount)} />
              <MiniPreviewItem label="Propinas" value={formatMoney(preview.tipsAmount)} />
              <MiniPreviewItem label="Adelantos" value={formatMoney(preview.advancesApplied)} />
              <MiniPreviewItem label="Pagos previos" value={formatMoney(preview.previousPaymentsApplied)} />
              <MiniPreviewItem label="Pendiente" value={formatMoney(preview.pendingAmount)} strong />
            </div>
          </div>
        )}

        <InputField
          label="Monto a pagar"
          value={amountPaid}
          onChange={handleAmountChange}
          type="number"
          step="0.01"
          prefix={getTenantCurrencySymbol()}
        />

        <div className="rounded-[26px] border border-neutral-200 bg-neutral-50 p-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <div className="text-sm font-black text-neutral-950">
                Distribuir por método de pago
              </div>
              <div className="mt-1 text-xs font-bold text-neutral-500">
                Puedes pagar una parte en efectivo y otra por Yape, Plin, tarjeta u otro método configurado.
              </div>
            </div>

            <button
              type="button"
              onClick={addPaymentRow}
              className="rounded-2xl border border-amber-200 bg-white px-4 py-3 text-xs font-black text-neutral-900 shadow-sm transition hover:bg-amber-50"
            >
              + Agregar método
            </button>
          </div>

          <div className="mt-4 space-y-3">
            {paymentRows.map((row, index) => (
              <div
                key={row.key}
                className="grid gap-3 rounded-2xl border border-neutral-200 bg-white p-3 sm:grid-cols-[1fr_1fr_auto]"
              >
                <SelectField
                  label={`Método ${index + 1}`}
                  value={normalizeMethod(row.method)}
                  onChange={(value) => updatePaymentRow(row.key, 'method', value)}
                  options={paymentOptions}
                />

                <InputField
                  label="Monto"
                  value={row.amount}
                  onChange={(value) => updatePaymentRow(row.key, 'amount', value)}
                  type="number"
                  step="0.01"
                  prefix={getTenantCurrencySymbol()}
                />

                <button
                  type="button"
                  onClick={() => removePaymentRow(row.key)}
                  disabled={paymentRows.length <= 1}
                  className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-xs font-black text-red-700 transition hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-40 sm:self-end"
                >
                  Quitar
                </button>
              </div>
            ))}
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <MiniPreviewItem label="Monto a pagar" value={formatMoney(amountToPay)} />
            <MiniPreviewItem label="Distribuido" value={formatMoney(distributedTotal)} />
            <MiniPreviewItem
              label={remainingToDistribute >= 0 ? 'Falta distribuir' : 'Excedente'}
              value={formatMoney(Math.abs(remainingToDistribute))}
              strong={Math.abs(remainingToDistribute) < 0.01}
            />
          </div>

          {Math.abs(remainingToDistribute) >= 0.01 && (
            <div className="mt-3 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-bold text-amber-800">
              La suma de los métodos debe coincidir con el monto a pagar antes de confirmar.
            </div>
          )}
        </div>

        <TextAreaField
          label="Observación"
          value={note}
          onChange={setNote}
          placeholder={`Ej. Pago semanal del ${labels.professionalSingular}`}
        />

        {errorMsg && <ErrorBox message={errorMsg} />}

        <button
          disabled={saving}
          className="w-full rounded-2xl bg-amber-400 px-5 py-4 font-black text-neutral-950 transition hover:scale-[1.01] disabled:opacity-60"
        >
          {saving ? 'Registrando pago...' : 'Confirmar pago'}
        </button>
      </form>
    </ModalShell>
  );
}

function MiniPreviewItem({ label, value, strong = false }) {
  return (
    <div
      className={`rounded-2xl border px-4 py-3 ${
        strong ? 'border-emerald-200 bg-emerald-50' : 'border-neutral-200 bg-white'
      }`}
    >
      <div className="text-xs font-bold text-neutral-500">{label}</div>
      <div className={`mt-1 text-lg font-black ${strong ? 'text-emerald-700' : 'text-neutral-950'}`}>
        {value}
      </div>
    </div>
  );
}

function CourtesySummarySection({ summary, labels = readBusinessLabels() }) {
  if (!summary?.count) return null;

  return (
    <section className="rounded-[32px] border border-amber-200 bg-amber-50 p-6 shadow-[0_16px_45px_rgba(15,23,42,0.05)]">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-700">
            {labels.courtesyPlural[0].toUpperCase() + labels.courtesyPlural.slice(1)} de hoy
          </div>
          <h3 className="mt-2 text-2xl font-black text-neutral-950">
            Servicios gratis registrados
          </h3>
          <p className="mt-1 max-w-2xl text-sm leading-6 text-amber-800/75">
            Controla cuantas {labels.courtesyPlural} se entregaron y su valor referencial por {labels.professionalSingular}.
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-2xl bg-white px-5 py-4 shadow-sm">
            <div className="text-xs font-black uppercase tracking-[0.16em] text-amber-700">
              Gratis
            </div>
            <div className="mt-1 text-2xl font-black text-neutral-950">
              {summary.count}
            </div>
          </div>
          <div className="rounded-2xl bg-white px-5 py-4 shadow-sm">
            <div className="text-xs font-black uppercase tracking-[0.16em] text-amber-700">
              Valor ref.
            </div>
            <div className="mt-1 text-2xl font-black text-neutral-950">
              {formatMoney(summary.referenceValue)}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {summary.byBarber.map((item) => (
          <div key={item.barberName} className="rounded-[24px] border border-amber-100 bg-white p-4 shadow-sm">
            <div className="text-sm font-black text-neutral-950">{item.barberName}</div>
            <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
              <div className="rounded-2xl bg-amber-50 px-3 py-2">
                <div className="text-xs font-bold text-amber-700">Cantidad</div>
                <div className="mt-1 font-black text-neutral-950">{item.count}</div>
              </div>
              <div className="rounded-2xl bg-neutral-50 px-3 py-2">
                <div className="text-xs font-bold text-neutral-500">Ref.</div>
                <div className="mt-1 font-black text-neutral-950">{formatMoney(item.referenceValue)}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function FiscalSaleCell({ document, onView }) {
  if (!document) return <div className="flex min-w-[145px] flex-col items-start gap-2">
    <span className="text-xs font-bold text-neutral-400">Sin comprobante</span>
    <button type="button" onClick={onView} className="rounded-xl border border-violet-200 bg-violet-50 px-3 py-2 text-xs font-black text-violet-700 hover:bg-violet-100">Emitir / recuperar</button>
  </div>;
  const accepted = document.status === 'ACCEPTED' || document.status === 'ACCEPTED_WITH_OBSERVATIONS';
  const label = `${document.documentType === 'INVOICE' ? 'Factura' : 'Boleta'} ${document.series || '}-${document.sequence || '}`;
  if (!accepted) return <button type="button" onClick={onView} className="rounded-xl bg-amber-50 px-3 py-2 text-xs font-black text-amber-700">{label} · {document.status}</button>;
  return <div className="flex min-w-[155px] flex-col items-start gap-2">
    <span className="rounded-full bg-emerald-100 px-3 py-1 text-[10px] font-black text-emerald-700">{label}</span>
    <details className="relative">
      <summary className="cursor-pointer list-none rounded-xl bg-neutral-950 px-3 py-2 text-xs font-black text-white">Imprimir ▾</summary>
      <div className="absolute right-0 z-30 mt-2 w-52 rounded-2xl border border-neutral-200 bg-white p-2 shadow-xl">
        <button type="button" onClick={() => openElectronicPdfForPrint(document)} className="w-full rounded-xl px-3 py-2 text-left text-xs font-black text-neutral-800 hover:bg-amber-50">Imprimir en ticketera</button>
        <button type="button" onClick={() => openElectronicPdfForPrint(document, null, { thermal: false })} className="w-full rounded-xl px-3 py-2 text-left text-xs font-black text-neutral-800 hover:bg-amber-50">Abrir PDF / impresora normal</button>
        <button type="button" onClick={onView} className="w-full rounded-xl px-3 py-2 text-left text-xs font-black text-violet-700 hover:bg-violet-50">Ver archivos y compartir</button>
      </div>
    </details>
  </div>;
}

function SalesSection({ sales, canManageSales, labels = readBusinessLabels(), onView, onEdit, onDelete, fiscalRefreshKey = 0 }) {
  const [invoicingAvailable, setInvoicingAvailable] = useState(false);
  const [fiscalBySale, setFiscalBySale] = useState({});
  const saleIdsKey = sales.map((sale) => saleIdOf(sale)).filter(Boolean).join(',');

  useEffect(() => {
    let active = true;
    async function loadFiscalState() {
      try {
        const access = await getElectronicInvoicingAccess();
        if (!active || !access?.available) { if (active) setInvoicingAvailable(false); return; }
        setInvoicingAvailable(true);
        const entries = await Promise.all(sales.map(async (sale) => {
          const saleId = saleIdOf(sale);
          if (!saleId) return [saleId, null];
          try { const documents = await getSaleElectronicDocuments(saleId); return [saleId, Array.isArray(documents) ? documents[0] || null : null]; }
          catch { return [saleId, null]; }
        }));
        if (active) setFiscalBySale(Object.fromEntries(entries));
      } catch { if (active) setInvoicingAvailable(false); }
    }
    loadFiscalState();
    return () => { active = false; };
  }, [saleIdsKey, fiscalRefreshKey]);
  return (
    <div className="rounded-[32px] border border-neutral-200 bg-white p-6 shadow-[0_16px_45px_rgba(15,23,42,0.05)]">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-600">
            Ventas
          </div>

          <h3 className="mt-2 text-2xl font-black text-neutral-950">
            Ventas de hoy
          </h3>

          <p className="mt-1 text-sm text-neutral-500">
            Ventas registradas en la sede seleccionada.
          </p>
        </div>

        <div className="rounded-2xl bg-neutral-100 px-4 py-3 text-sm font-black text-neutral-700">
          {sales.length} venta{sales.length === 1 ? '' : 's'}
        </div>
      </div>

      <div className="mt-6 overflow-hidden rounded-[26px] border border-neutral-200">
        <table className="w-full text-left text-sm">
          <thead className="bg-[linear-gradient(135deg,#090909_0%,#111827_100%)] text-white">
            <tr>
              <th className="px-5 py-4 font-black">Cliente</th>
              <th className="px-5 py-4 font-black">{labels.professionalSingular[0].toUpperCase() + labels.professionalSingular.slice(1)}</th>
              <th className="px-5 py-4 font-black">Método</th>
              <th className="px-5 py-4 font-black">Total</th>
              <th className="px-5 py-4 font-black">Fecha</th>
              {invoicingAvailable && <th className="px-5 py-4 font-black">Comprobante</th>}
              <th className="px-5 py-4 font-black text-right">Acciones</th>
            </tr>
          </thead>

          <tbody>
            {sales.length === 0 ? (
              <tr>
                <td className="px-5 py-6 text-neutral-500" colSpan={invoicingAvailable ? 7 : 6}>
                  Aún no hay ventas registradas hoy.
                </td>
              </tr>
            ) : (
              sales.map((sale) => (
                <tr key={saleIdOf(sale)} className="border-t border-neutral-200 transition hover:bg-amber-50/50">
                  <td className="px-5 py-5">
                    <div className="font-black text-neutral-950">
                      {sale.customerName || 'Cliente ocasional'}
                    </div>
                    <div className="mt-1 text-xs text-neutral-400">
                      ID: {saleIdOf(sale) || '-'}
                    </div>
                  </td>

                  <td className="px-5 py-5 font-bold text-neutral-700">
                    {saleBarberName(sale)}
                  </td>

                  <td className="px-5 py-5">
                    <div className="font-black text-neutral-800">
                      {methodLabel(sale.metodoPago)}
                    </div>
                    <div className="mt-1 max-w-[240px] text-xs font-bold leading-5 text-neutral-500">
                      {salePaymentSummary(sale)}
                    </div>
                  </td>

                  <td className="px-5 py-5 font-black text-emerald-700">
                    {formatMoney(sale.total)}
                  </td>

                  <td className="px-5 py-5 text-xs font-bold text-neutral-500">
                    {formatDateTime(saleDateOf(sale))}
                  </td>

                  {invoicingAvailable && <td className="px-5 py-5">                     <FiscalSaleCell document={fiscalBySale[saleIdOf(sale)]} onView={() => onView(sale)} />                   </td>}

                  <td className="px-5 py-5">
                    <div className="flex flex-wrap justify-end gap-2">
                      <button
                        onClick={() => onView(sale)}
                        className="rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-black text-emerald-700 hover:bg-emerald-100"
                      >
                        Ver detalle
                      </button>
                      {canManageSales && !(fiscalBySale[saleIdOf(sale)]?.status === 'ACCEPTED' || fiscalBySale[saleIdOf(sale)]?.status === 'ACCEPTED_WITH_OBSERVATIONS') && (
                        <>
                          <button
                            onClick={() => onEdit(sale)}
                            className="rounded-xl border border-neutral-200 bg-white px-3 py-2 text-xs font-black text-neutral-700 hover:bg-neutral-50"
                          >
                            Editar
                          </button>
                          <button
                            onClick={() => onDelete(sale)}
                            className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs font-black text-red-700 hover:bg-red-100"
                          >
                            Eliminar
                          </button>
                        </>
                      )}
                      {canManageSales && (fiscalBySale[saleIdOf(sale)]?.status === 'ACCEPTED' || fiscalBySale[saleIdOf(sale)]?.status === 'ACCEPTED_WITH_OBSERVATIONS') && (
                        <span className="rounded-xl bg-emerald-50 px-3 py-2 text-[10px] font-black text-emerald-700">Protegida por SUNAT</span>
                      )}
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}


function saleItemsOf(sale) {
  return Array.isArray(sale?.items)
    ? sale.items
    : Array.isArray(sale?.saleItems)
      ? sale.saleItems
      : [];
}

function saleItemName(item) {
  return String(
    item?.serviceName ??
      item?.serviceNombre ??
      item?.productName ??
      item?.nombreItem ??
      item?.name ??
      'Item'
  );
}

function saleItemTypeLabel(item) {
  if (item?.productId || item?.productName) return 'Producto';
  return 'Servicio';
}

function saleItemBarberName(item) {
  return String(
    item?.barberUserName ??
      item?.barberName ??
      item?.barbero ??
      '-'
  );
}

function saleItemQuantity(item) {
  return Number(item?.cantidad ?? item?.quantity ?? 1) || 1;
}

function saleItemUnitPrice(item) {
  return Number(item?.precioUnitario ?? item?.unitPrice ?? 0) || 0;
}

function saleItemTotal(item) {
  const direct = Number(item?.subtotal ?? item?.total ?? 0);
  if (direct > 0) return direct;
  return saleItemQuantity(item) * saleItemUnitPrice(item);
}

function DetailMetric({ label, value, tone = 'default', helper }) {
  const styles = {
    default: 'border-neutral-200 bg-white text-neutral-950',
    gold: 'border-amber-200 bg-amber-50 text-amber-800',
    green: 'border-emerald-200 bg-emerald-50 text-emerald-800',
    red: 'border-red-200 bg-red-50 text-red-800',
    dark: 'border-neutral-900 bg-neutral-950 text-white',
  };

  return (
    <div className={`rounded-[22px] border p-4 ${styles[tone] || styles.default}`}>
      <div className={tone === 'dark' ? 'text-xs font-black uppercase tracking-[0.14em] text-white/45' : 'text-xs font-black uppercase tracking-[0.14em] text-neutral-500'}>
        {label}
      </div>
      <div className={tone === 'dark' ? 'mt-2 text-xl font-black text-white' : 'mt-2 text-xl font-black'}>
        {value}
      </div>
      {helper && (
        <div className={tone === 'dark' ? 'mt-1 text-xs font-bold text-white/45' : 'mt-1 text-xs font-bold text-neutral-500'}>
          {helper}
        </div>
      )}
    </div>
  );
}

async function openElectronicPdfForPrint(document, preparedPopup = null, { thermal = true } = {}) {
  const popup = preparedPopup || window.open('', '_blank');
  if (!popup) {
    window.alert('El navegador bloqueó la ventana. Permite ventanas emergentes para imprimir el comprobante.');
    return false;
  }
  popup.document.write('<title>Preparando comprobante</title><p style="font-family:system-ui;padding:32px">Preparando PDF oficial...</p>');
  try {
    const files = await getElectronicDocumentFiles(document.id);
    if (thermal) {
      try {
        const thermalResult = await printElectronicPdfOnThermal(files);
        if (thermalResult?.printed) {
          popup.close();
          return true;
        }
      } catch (thermalError) {
        console.error('No se pudo imprimir el comprobante oficial por QZ Tray:', thermalError);
        window.alert(`No se pudo imprimir directamente en la ticketera: ${thermalError.message || 'error de impresora'}. Se abrirá el PDF para impresión manual.`);
      }
    }
    if (files.documentUrl) { popup.location.href = files.documentUrl; return true; }
    if (files.pdfBase64) {
      const binary = window.atob(files.pdfBase64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      const url = URL.createObjectURL(new Blob([bytes], { type: 'application/pdf' }));
      popup.location.href = url;
      window.setTimeout(() => URL.revokeObjectURL(url), 120000);
      return true;
    }
    throw new Error('El PDF aún no está disponible.');
  } catch (reason) {
    popup.close();
    window.alert(reason.message || 'No se pudo abrir el PDF.');
    return false;
  }
}

function ElectronicDocumentPanel({ saleId, branchId, onChanged, initialDraft = null, initialError = '' }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState(initialError);
  const [issueType, setIssueType] = useState(initialDraft?.documentType || 'RECEIPT');
  const [issueDocument, setIssueDocument] = useState(initialDraft?.receiverDocumentNumber || '');
  const [issueName, setIssueName] = useState(initialDraft?.receiverName || '');
  const [issueAddress, setIssueAddress] = useState(initialDraft?.receiverAddress || '');
  const [issueEmail, setIssueEmail] = useState(initialDraft?.receiverEmail || '');
  const [printAfterIssue, setPrintAfterIssue] = useState(initialDraft?.printAfterIssue !== false);
  async function load() {
    setLoading(true);
    try { const data = await getSaleElectronicDocuments(saleId); setDocuments(Array.isArray(data) ? data : []); }
    catch (reason) { setError(reason.message || 'No se pudo consultar el comprobante.'); }
    finally { setLoading(false); }
  }
  useEffect(() => { load(); }, [saleId]);
  async function download(document, type) {
    setWorking(true); setError('');
    try {
      const files = await getElectronicDocumentFiles(document.id);
      const number = `${document.series || 'CPE'}-${document.sequence || document.id}`;
      if (type === 'SHARE') {
        if (!files.documentUrl) throw new Error('El enlace público todavía no está disponible.');
        if (navigator.share) await navigator.share({ title: `Comprobante ${number}`, text: 'Comprobante electrónico', url: files.documentUrl });
        else { await navigator.clipboard.writeText(files.documentUrl); window.alert('Enlace del comprobante copiado.'); }
      } else if (type === 'THERMAL') {
        const result = await printElectronicPdfOnThermal(files);
        if (!result?.printed) throw new Error('Configura y conecta la ticketera en Configuración > Impresora.');
        window.alert(`Comprobante enviado a ${result.printerName} en papel de ${result.paperWidth} mm.`);
      } else if (type === 'PDF') {
        if (files.pdfBase64) downloadBase64File(files.pdfBase64, 'application/pdf', `${number}.pdf`);
        else if (files.documentUrl) window.open(files.documentUrl, '_blank', 'noopener,noreferrer');
        else throw new Error('El proveedor todavía no devolvió el PDF.');
      } else if (type === 'XML') {
        if (!downloadBase64File(files.xmlBase64, 'application/xml', `${number}.xml`)) throw new Error('XML todavía no disponible.');
      } else if (!downloadBase64File(files.cdrBase64, 'application/zip', `${number}-CDR.zip`)) throw new Error('CDR todavía no disponible.');
    } catch (reason) { setError(reason.message || 'No se pudo descargar el archivo.'); }
    finally { setWorking(false); }
  }
  async function update(document) {
    setWorking(true); setError('');
    try {
      if (document.status === 'ERROR') await retryElectronicDocument(document.id); else await refreshElectronicDocument(document.id);
      await load();
      onChanged?.();
    } catch (reason) { setError(reason.message || 'No se pudo actualizar el comprobante.'); }
    finally { setWorking(false); }
  }
  async function issueMissingDocument() {
    const digits = issueDocument.replace(/\D/g, '');
    if (!branchId) { setError('No se pudo identificar la sede de esta venta.'); return; }
    if (issueType === 'RECEIPT' && digits.length !== 8) { setError('Ingresa un DNI válido de 8 dígitos.'); return; }
    if (issueType === 'INVOICE' && digits.length !== 11) { setError('Ingresa un RUC válido de 11 dígitos.'); return; }
    if (!issueName.trim()) { setError('Ingresa el nombre o razón social.'); return; }
    if (issueType === 'INVOICE' && !issueAddress.trim()) { setError('La dirección fiscal es obligatoria.'); return; }
    const preparedPrintPopup = printAfterIssue ? window.open('', '_blank') : null;
    if (preparedPrintPopup) {
      preparedPrintPopup.document.write('<title>Preparando comprobante</title><p style="font-family:system-ui;padding:32px">Emitiendo comprobante y preparando PDF oficial...</p>');
    }
    setWorking(true); setError('');
    try {
      const issuedDocument = await issueElectronicDocument({ saleId, branchId, documentType: issueType, receiverDocumentType: issueType === 'INVOICE' ? '6' : '1', receiverDocumentNumber: digits, receiverName: issueName.trim(), receiverAddress: issueAddress.trim() || '-', receiverEmail: issueEmail.trim() || null });
      const accepted = issuedDocument?.status === 'ACCEPTED' || issuedDocument?.status === 'ACCEPTED_WITH_OBSERVATIONS';
      if (accepted && printAfterIssue) await openElectronicPdfForPrint(issuedDocument, preparedPrintPopup);
      else preparedPrintPopup?.close();
      await load();
      onChanged?.();
    } catch (reason) {
      preparedPrintPopup?.close();
      setError(reason.message || 'No se pudo emitir el comprobante.');
    } finally { setWorking(false); }
  }

  if (loading) return <div className="rounded-[28px] border border-violet-200 bg-violet-50 p-5 text-sm font-black text-violet-700">Consultando comprobante electrónico...</div>;
  if (documents.length === 0) return <div className="rounded-[28px] border border-violet-200 bg-violet-50 p-5">
    <div className="font-black text-neutral-900">Sin comprobante electrónico</div>
    <p className="mt-1 text-sm font-semibold text-neutral-600">Puedes emitirlo sobre esta misma venta sin duplicar caja.</p>
    <div className="mt-4 grid grid-cols-2 gap-2">
      {[['RECEIPT','Boleta'],['INVOICE','Factura']].map(([value,label]) => <button key={value} type="button" onClick={() => { setIssueType(value); setIssueDocument(''); }} className={`rounded-xl px-3 py-3 text-xs font-black ${issueType === value ? 'bg-neutral-950 text-white' : 'border border-violet-200 bg-white text-neutral-700'}`}>{label}</button>)}
    </div>
    <div className="mt-3 grid gap-3 sm:grid-cols-2">
      <InputField label={issueType === 'INVOICE' ? 'RUC' : 'DNI'} value={issueDocument} onChange={(value) => setIssueDocument(value.replace(/\D/g, '').slice(0, issueType === 'INVOICE' ? 11 : 8))} />
      <InputField label={issueType === 'INVOICE' ? 'Razón social' : 'Nombre del cliente'} value={issueName} onChange={setIssueName} />
      {issueType === 'INVOICE' && <div className="sm:col-span-2"><InputField label="Dirección fiscal" value={issueAddress} onChange={setIssueAddress} /></div>}
      <div className="sm:col-span-2"><InputField label="Correo opcional" value={issueEmail} onChange={setIssueEmail} type="email" /></div>
    </div>
    <label className="mt-3 flex items-center justify-between gap-4 rounded-xl border border-violet-200 bg-white px-4 py-3">
      <span><span className="block text-sm font-black text-neutral-900">Imprimir al emitir</span><span className="mt-1 block text-xs font-bold text-neutral-500">Abre el PDF oficial aceptado por SUNAT.</span></span>
      <input type="checkbox" checked={printAfterIssue} onChange={(event) => setPrintAfterIssue(event.target.checked)} className="h-6 w-6 shrink-0 accent-violet-600" />
    </label>
    {error && <div className="mt-3 rounded-xl bg-red-50 p-3 text-xs font-bold text-red-700"><div className="font-black">No se pudo emitir al guardar la venta</div><div className="mt-1">{error}</div></div>}
    <button disabled={working} type="button" onClick={issueMissingDocument} className="mt-4 w-full rounded-xl bg-violet-600 px-4 py-3 text-sm font-black text-white disabled:opacity-50">{working ? 'Emitiendo...' : `Emitir ${issueType === 'INVOICE' ? 'factura' : 'boleta'} ahora`}</button>
  </div>;
  return <div className="space-y-3">{documents.map((document) => {
    const accepted = document.status === 'ACCEPTED' || document.status === 'ACCEPTED_WITH_OBSERVATIONS';
    return <div key={document.id} className={`rounded-[28px] border p-5 ${accepted ? 'border-emerald-200 bg-emerald-50' : 'border-amber-200 bg-amber-50'}`}>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between"><div>
        <div className="text-xs font-black uppercase tracking-[0.18em] text-violet-700">Comprobante electrónico</div>
        <h3 className="mt-2 text-xl font-black text-neutral-950">{document.documentType === 'INVOICE' ? 'Factura' : 'Boleta'} {document.series}-{document.sequence}</h3>
        <p className="mt-1 text-sm font-bold text-neutral-600">Estado: {accepted ? 'Aceptado por SUNAT' : document.status}</p>
        {document.sunatDescription && <p className="mt-1 text-xs font-semibold text-neutral-500">{document.sunatDescription}</p>}
      </div><span className={`rounded-full px-3 py-2 text-xs font-black ${accepted ? 'bg-emerald-600 text-white' : 'bg-amber-500 text-neutral-950'}`}>{document.status}</span></div>
      {error && <div className="mt-3 rounded-xl bg-red-50 p-3 text-xs font-bold text-red-700">{error}</div>}
      <div className="mt-4 flex flex-wrap gap-2">
        {accepted && <><button disabled={working} type="button" onClick={() => download(document, 'THERMAL')} className="rounded-xl bg-neutral-950 px-4 py-3 text-xs font-black text-white disabled:opacity-50">Imprimir en ticketera</button>
        <button disabled={working} type="button" onClick={() => download(document, 'PDF')} className="rounded-xl border border-neutral-300 bg-white px-4 py-3 text-xs font-black text-neutral-800 disabled:opacity-50">Ver / descargar PDF</button>
        <button disabled={working} type="button" onClick={() => download(document, 'SHARE')} className="rounded-xl border border-violet-300 bg-violet-50 px-4 py-3 text-xs font-black text-violet-800 disabled:opacity-50">Compartir</button>
        <button disabled={working} type="button" onClick={() => download(document, 'XML')} className="rounded-xl border border-neutral-300 bg-white px-4 py-3 text-xs font-black text-neutral-800 disabled:opacity-50">Descargar XML</button>
        <button disabled={working} type="button" onClick={() => download(document, 'CDR')} className="rounded-xl border border-neutral-300 bg-white px-4 py-3 text-xs font-black text-neutral-800 disabled:opacity-50">Descargar CDR</button></>}
        {!accepted && <button disabled={working} type="button" onClick={() => update(document)} className="rounded-xl bg-amber-400 px-4 py-3 text-xs font-black text-neutral-950 disabled:opacity-50">{document.status === 'ERROR' ? 'Reintentar' : 'Actualizar estado'}</button>}
      </div>
    </div>;
  })}</div>;
}

function SaleDetailModal({ sale, branchId, onClose, onElectronicDocumentChanged, labels = readBusinessLabels() }) {
  const payments = salePaymentsOf(sale);
  const items = saleItemsOf(sale);
  const subtotal = Number(sale?.subtotal ?? 0);
  const discount = Number(sale?.discount ?? 0);
  const tipAmount = Number(sale?.tipAmount ?? 0);
  const total = Number(sale?.total ?? 0);
  const cashReceived = Number(sale?.cashReceived ?? 0);
  const changeAmount = Number(sale?.changeAmount ?? 0);
  const saleId = saleIdOf(sale);

  return (
    <ModalShell
      title={`Detalle de venta #${saleId || '-'}`}
      subtitle={sale?.customerName || 'Cliente ocasional'}
      onClose={onClose}
      maxWidth="max-w-5xl"
    >
      <div className="space-y-5">
        <ElectronicDocumentPanel saleId={saleId} branchId={branchId ?? sale?.branchId ?? sale?.branch?.id ?? sale?.sedeId} onChanged={onElectronicDocumentChanged} initialDraft={sale?.pendingElectronicDocument} initialError={sale?.electronicDocumentError || ''} />
        <div className="rounded-[28px] border border-neutral-200 bg-[linear-gradient(135deg,#F8FAFC_0%,#FFFFFF_70%)] p-5">
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <DetailMetric label="Cliente" value={sale?.customerName || 'Cliente ocasional'} helper={sale?.customerId ? `ID cliente: ${sale.customerId}` : 'Sin cliente registrado'} />
            <DetailMetric label={labels.professionalDisplay} value={saleBarberName(sale)} helper="Principal o primer item" />
            <DetailMetric label="Fecha" value={formatDateTime(saleDateOf(sale))} helper="Fecha de venta" />
            <DetailMetric label="Método" value={methodLabel(sale?.metodoPago)} helper={payments.length > 1 ? 'Pago mixto' : 'Pago único'} tone={payments.length > 1 ? 'gold' : 'default'} />
          </div>
        </div>

        <div className="grid gap-5 xl:grid-cols-[1.15fr_0.85fr]">
          <div className="rounded-[28px] border border-neutral-200 bg-white p-5 shadow-[0_12px_32px_rgba(15,23,42,0.045)]">
            <div className="flex items-end justify-between gap-3">
              <div>
                <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-600">
                  Items vendidos
                </div>
                <div className="mt-1 text-sm font-bold text-neutral-500">
                  Servicios y productos incluidos en esta venta.
                </div>
              </div>
              <div className="rounded-full bg-neutral-100 px-3 py-1 text-xs font-black text-neutral-600">
                {items.length} item{items.length === 1 ? '' : 's'}
              </div>
            </div>

            <div className="mt-4 space-y-3">
              {items.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-neutral-300 bg-neutral-50 p-5 text-sm font-bold text-neutral-500">
                  Esta venta no trajo detalle de items en la respuesta.
                </div>
              ) : (
                items.map((item, index) => (
                  <div key={`${saleItemName(item)}-${index}`} className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div className="min-w-0">
                        <div className="truncate text-base font-black text-neutral-950">
                          {saleItemName(item)}
                        </div>
                        <div className="mt-1 text-xs font-bold text-neutral-500">
                          {saleItemTypeLabel(item)} · {labels.professionalDisplay}: {saleItemBarberName(item)}
                        </div>
                      </div>

                      <div className="text-left sm:text-right">
                        <div className="text-base font-black text-neutral-950">
                          {formatMoney(saleItemTotal(item))}
                        </div>
                        <div className="mt-1 text-xs font-bold text-neutral-500">
                          {saleItemQuantity(item)} x {formatMoney(saleItemUnitPrice(item))}
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="space-y-5">
            <div className="rounded-[28px] border border-neutral-200 bg-white p-5 shadow-[0_12px_32px_rgba(15,23,42,0.045)]">
              <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-600">
                Métodos de pago
              </div>
              <div className="mt-4 space-y-3">
                {payments.length === 0 ? (
                  <div className="rounded-2xl bg-neutral-50 p-4 text-sm font-bold text-neutral-500">
                    {methodLabel(sale?.metodoPago)} · {formatMoney(total)}
                  </div>
                ) : (
                  payments.map((payment, index) => (
                    <div key={`${payment.method}-${index}`} className="flex items-center justify-between gap-3 rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4">
                      <div>
                        <div className="font-black text-neutral-950">
                          {methodLabel(payment.method)}
                        </div>
                        <div className="text-xs font-bold text-neutral-500">
                          Método {index + 1}
                        </div>
                      </div>
                      <div className="text-lg font-black text-emerald-700">
                        {formatMoney(payment.amount)}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="rounded-[28px] border border-neutral-200 bg-white p-5 shadow-[0_12px_32px_rgba(15,23,42,0.045)]">
              <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-600">
                Resumen económico
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <DetailMetric label="Subtotal" value={formatMoney(subtotal)} />
                <DetailMetric label="Descuento" value={formatMoney(discount)} tone={discount > 0 ? 'red' : 'default'} />
                <DetailMetric label="Propina" value={formatMoney(tipAmount)} tone={tipAmount > 0 ? 'green' : 'default'} helper={sale?.tipBarberUserName ? `Para ${sale.tipBarberUserName}` : null} />
                <DetailMetric label="Total" value={formatMoney(total)} tone="dark" />
                <DetailMetric label="Recibido" value={formatMoney(cashReceived)} helper="Solo efectivo recibido" />
                <DetailMetric label="Vuelto" value={formatMoney(changeAmount)} tone={changeAmount > 0 ? 'green' : 'default'} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </ModalShell>
  );
}

function EditSaleModal({ branch, sale, paymentMethods = DEFAULT_PAYMENT_METHODS, session = null, onClose, onSaved }) {
  const salePayments = salePaymentsOf(sale).filter(
    (payment) => normalizeMethod(payment.method) !== 'DEPOSIT_APPLIED'
  );

  const initialPayments = salePayments.length > 0
    ? salePayments.map((payment) => createPaymentDraft(normalizeMethod(payment.method), payment.amount))
    : [createPaymentDraft(normalizeMethod(sale?.metodoPago || 'CASH'), Number(sale?.total || 0))];

  const [subtotal, setSubtotal] = useState(String(Number(sale?.subtotal ?? sale?.total ?? 0).toFixed(2)));
  const [discount, setDiscount] = useState(String(Number(sale?.discount ?? 0).toFixed(2)));
  const [total, setTotal] = useState(String(Number(sale?.total ?? 0).toFixed(2)));
  const [cashReceived, setCashReceived] = useState(String(Number(sale?.cashReceived ?? sale?.total ?? 0).toFixed(2)));
  const [auditReason, setAuditReason] = useState('');
  const initialItems = saleItemsOf(sale).map((item, index) => ({
    key: String(item.saleItemId || item.id || `item-${index}`),
    saleItemId: item.saleItemId || item.id,
    name: saleItemName(item),
    type: saleItemTypeLabel(item),
    quantity: saleItemQuantity(item),
    barberUserId: item.barberUserId ? String(item.barberUserId) : '',
    precioUnitario: String(Number(saleItemUnitPrice(item) || 0).toFixed(2)),
  }));
  const [payments, setPayments] = useState(initialPayments);
  const [items, setItems] = useState(initialItems);
  const [barbers, setBarbers] = useState([]);
  const [loadingBarbers, setLoadingBarbers] = useState(true);
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const paymentOptions = paymentSelectOptions(paymentMethods);

  useEffect(() => {
    let alive = true;
    setLoadingBarbers(true);
    getCashBarbers(branch?.id)
      .then((data) => { if (alive) setBarbers(mergeOwnerIntoBarbers(data, session)); })
      .catch(() => { if (alive) setBarbers([]); })
      .finally(() => { if (alive) setLoadingBarbers(false); });
    return () => { alive = false; };
  }, [branch?.id, session]);

  const totalNumber = roundMoney(parseMoneyInput(total));
  const paymentPayloads = payments
    .map((payment) => ({
      method: normalizeMethod(payment.method),
      amount: roundMoney(parseMoneyInput(payment.amount)),
    }))
    .filter((payment) => payment.amount > 0);

  const paymentsTotal = roundMoney(paymentPayloads.reduce((sum, payment) => sum + payment.amount, 0));
  const remainingPayment = roundMoney(totalNumber - paymentsTotal);
  const cashPaymentAmount = roundMoney(
    paymentPayloads
      .filter((payment) => normalizeMethod(payment.method) === 'CASH')
      .reduce((sum, payment) => sum + payment.amount, 0)
  );
  const cashReceivedNumber = roundMoney(parseMoneyInput(cashReceived));
  const changeAmount = cashPaymentAmount > 0
    ? roundMoney(Math.max(0, cashReceivedNumber - cashPaymentAmount))
    : 0;

  function updateItem(index, patch) {
    setItems((prev) => prev.map((item, currentIndex) => (
      currentIndex === index ? { ...item, ...patch } : item
    )));
  }

  function updatePayment(index, patch) {
    setPayments((prev) => prev.map((payment, currentIndex) => (
      currentIndex === index ? { ...payment, ...patch } : payment
    )));
  }

  function addPaymentMethod() {
    setPayments((prev) => [...prev, createPaymentDraft(defaultExtraPaymentMethod(paymentMethods), 0)]);
  }

  function removePaymentMethod(index) {
    setPayments((prev) => prev.filter((_, currentIndex) => currentIndex !== index));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMsg('');

    const saleId = saleIdOf(sale);
    if (!saleId) {
      setErrorMsg('No se encontró el ID de la venta.');
      return;
    }

    if (totalNumber < 0) {
      setErrorMsg('El total no puede ser negativo.');
      return;
    }

    if (totalNumber > 0 && paymentPayloads.length === 0) {
      setErrorMsg('Agrega al menos un método de pago.');
      return;
    }

    if (Math.abs(remainingPayment) > 0.009) {
      setErrorMsg(
        remainingPayment > 0
          ? `Falta cobrar ${formatMoney(remainingPayment)}.`
          : `Los pagos superan el total por ${formatMoney(Math.abs(remainingPayment))}.`
      );
      return;
    }

    if (cashPaymentAmount > 0 && cashReceivedNumber + 0.009 < cashPaymentAmount) {
      setErrorMsg('El efectivo recibido no puede ser menor al monto pagado en efectivo.');
      return;
    }

    if (!auditReason.trim()) {
      setErrorMsg('Escribe el motivo de auditoria para editar esta venta.');
      return;
    }

    const mainPaymentMethod = totalNumber === 0
      ? 'FREE'
      : paymentPayloads.length > 1
        ? 'MIXED'
        : paymentPayloads[0]?.method || 'CASH';

    setSaving(true);

    try {
      await updateCashSale({
        branchId: branch.id,
        saleId,
        customerId: sale.customerId ?? null,
        metodoPago: mainPaymentMethod,
        subtotal: parseMoneyInput(subtotal),
        discount: parseMoneyInput(discount),
        total: totalNumber,
        cashReceived: cashPaymentAmount > 0 ? cashReceivedNumber : totalNumber,
        changeAmount,
        auditReason: auditReason.trim(),
        payments: totalNumber === 0 ? [] : paymentPayloads,
        items: items.map((item) => ({
          saleItemId: item.saleItemId,
          barberUserId: item.barberUserId || null,
          precioUnitario: parseMoneyInput(item.precioUnitario),
        })),
      });

      onSaved();
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo actualizar la venta.');
    } finally {
      setSaving(false);
    }
  }

  return (
    <ModalShell title="Editar venta" subtitle={branch?.name || 'Sede'} onClose={onClose} maxWidth="max-w-5xl">
      <form onSubmit={handleSubmit} className="space-y-5">
        <div>
          <label className="mb-2 block text-xs font-black uppercase tracking-[0.16em] text-neutral-500">Motivo de auditoria</label>
          <textarea
            value={auditReason}
            onChange={(event) => setAuditReason(event.target.value)}
            rows={3}
            placeholder="Ej. Corrección de profesional, valor o método de pago."
            className="w-full rounded-3xl border border-amber-200 bg-amber-50/60 px-4 py-3 text-sm font-bold text-neutral-900 outline-none transition focus:border-amber-400 focus:bg-white"
          />
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          <InputField label="Subtotal" value={subtotal} onChange={setSubtotal} type="number" step="0.01" prefix={getTenantCurrencySymbol()} />
          <InputField label="Descuento" value={discount} onChange={setDiscount} type="number" step="0.01" prefix={getTenantCurrencySymbol()} />
          <InputField label="Total" value={total} onChange={setTotal} type="number" step="0.01" prefix={getTenantCurrencySymbol()} />
        </div>

        <div className="rounded-[28px] border border-neutral-200 bg-white p-5">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <div className="text-xs font-black uppercase tracking-[0.20em] text-amber-600">
                Métodos de pago
              </div>
              <p className="mt-1 text-sm font-semibold text-neutral-500">
                Puedes editar pagos mixtos sin afectar Flutter. La suma debe coincidir con el total.
              </p>
            </div>

            <button
              type="button"
              onClick={addPaymentMethod}
              className="rounded-2xl border border-neutral-200 bg-neutral-950 px-4 py-3 text-sm font-black text-white transition hover:scale-[1.01]"
            >
              + Agregar método
            </button>
          </div>

          <div className="mt-4 space-y-3">
            {payments.map((payment, index) => (
              <div key={payment.key} className="rounded-[24px] border border-neutral-200 bg-neutral-50 p-4">
                <div className="grid gap-3 md:grid-cols-[1fr_1fr_auto] md:items-end">
                  <SelectField
                    label={`Método ${index + 1}`}
                    value={payment.method}
                    onChange={(value) => updatePayment(index, { method: value })}
                    options={paymentOptions}
                  />

                  <InputField
                    label="Monto"
                    value={payment.amount}
                    onChange={(value) => updatePayment(index, { amount: value })}
                    type="number"
                    step="0.01"
                    prefix={getTenantCurrencySymbol()}
                  />

                  <button
                    type="button"
                    onClick={() => removePaymentMethod(index)}
                    disabled={payments.length <= 1}
                    className="rounded-2xl border border-red-200 bg-red-50 px-4 py-4 text-sm font-black text-red-700 transition hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Quitar
                  </button>
                </div>
              </div>
            ))}
          </div>

          {cashPaymentAmount > 0 && (
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <InputField
                label="Efectivo recibido"
                value={cashReceived}
                onChange={setCashReceived}
                type="number"
                step="0.01"
                prefix={getTenantCurrencySymbol()}
              />
              <StatCard title="Vuelto" value={formatMoney(changeAmount)} tone={changeAmount > 0 ? 'green' : 'default'} />
            </div>
          )}

          <div className="mt-4 grid gap-3 md:grid-cols-3">
            <StatCard title="Total venta" value={formatMoney(totalNumber)} />
            <StatCard title="Pagado" value={formatMoney(paymentsTotal)} tone={Math.abs(remainingPayment) <= 0.009 ? 'green' : 'gold'} />
            <StatCard
              title={remainingPayment >= 0 ? 'Falta' : 'Sobra'}
              value={formatMoney(Math.abs(remainingPayment))}
              tone={Math.abs(remainingPayment) <= 0.009 ? 'green' : 'red'}
            />
          </div>
        </div>

        <div className="rounded-[28px] border border-neutral-200 bg-white p-5">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <div className="text-xs font-black uppercase tracking-[0.20em] text-amber-600">
                Items de la venta
              </div>
              <p className="mt-1 text-sm font-semibold text-neutral-500">
                Cambia el profesional asignado a cada servicio o producto registrado.
              </p>
            </div>
            {loadingBarbers && (
              <div className="text-xs font-black text-neutral-400">Cargando profesionales...</div>
            )}
          </div>

          <div className="mt-4 space-y-3">
            {items.length === 0 ? (
              <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 text-sm font-black text-neutral-500">
                Esta venta no tiene items editables.
              </div>
            ) : (
              items.map((item, index) => (
                <div key={item.key} className="rounded-[24px] border border-neutral-200 bg-neutral-50 p-4">
                  <div className="grid gap-3 lg:grid-cols-[1.2fr_1fr_180px] lg:items-end">
                    <div>
                      <div className="text-sm font-black text-neutral-950">{item.name}</div>
                      <div className="mt-1 text-xs font-bold text-neutral-500">
                        {item.type} - {item.quantity} unidad{Number(item.quantity || 0) === 1 ? '' : 'es'}
                      </div>
                    </div>

                    <SelectField
                      label="Profesional"
                      value={item.barberUserId}
                      onChange={(value) => updateItem(index, { barberUserId: value })}
                      options={[
                        { value: '', label: 'Sin profesional' },
                        ...barbers.map((barber) => ({
                          value: String(barber.id),
                          label: barber.name,
                        })),
                      ]}
                    />

                    <InputField
                      label="Precio unitario"
                      value={item.precioUnitario}
                      onChange={(value) => updateItem(index, { precioUnitario: value })}
                      type="number"
                      step="0.01"
                      prefix={getTenantCurrencySymbol()}
                    />
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-800">
          Este ajuste modifica la venta registrada. Si envías métodos de pago, el backend reemplazará los pagos anteriores de esta venta.
        </div>

        {errorMsg && <ErrorBox message={errorMsg} />}

        <button
          disabled={saving}
          className="w-full rounded-2xl bg-amber-400 px-5 py-4 font-black text-neutral-950 transition hover:scale-[1.01] disabled:opacity-60"
        >
          {saving ? 'Guardando...' : 'Guardar cambios'}
        </button>
      </form>
    </ModalShell>
  );
}


function AppointmentSaleBanner({ appointment, isOpen, onOpenSale, onDismiss, labels = readBusinessLabels() }) {
  if (!appointment) return null;

  return (
    <div className="rounded-[32px] border border-amber-200 bg-[linear-gradient(135deg,#FFFBEB_0%,#FFFFFF_75%)] p-5 shadow-[0_16px_42px_rgba(15,23,42,0.06)]">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div className="flex gap-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-neutral-950 text-2xl text-white">
            ✂️
          </div>

          <div>
            <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-700">
              Cita enviada desde agenda
            </div>
            <h3 className="mt-1 text-2xl font-black text-neutral-950">
              Atendiendo cita #{appointment.appointmentId}
            </h3>
            <div className="mt-2 flex flex-wrap gap-2 text-sm font-bold text-neutral-600">
              <span>Cliente: {appointment.customerName || 'Cliente'}</span>
              <span>•</span>
              <span>Servicio: {appointment.serviceName || 'Servicio'}</span>
              <span>•</span>
              <span>{labels.professionalDisplay}: {appointment.barberName || labels.professionalDisplay}</span>
              {appointment.hora && (
                <>
                  <span>•</span>
                  <span>{appointment.hora}{appointment.horaFin ? ` - ${appointment.horaFin}` : ''}</span>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={onOpenSale}
            disabled={!isOpen}
            className="rounded-2xl bg-neutral-950 px-5 py-4 text-sm font-black text-white transition hover:scale-[1.01] disabled:bg-neutral-200 disabled:text-neutral-500"
          >
            Cobrar cita
          </button>

          <button
            type="button"
            onClick={onDismiss}
            className="rounded-2xl border border-neutral-200 bg-white px-5 py-4 text-sm font-black text-neutral-700 hover:bg-neutral-50"
          >
            Quitar aviso
          </button>
        </div>
      </div>

      {!isOpen && (
        <div className="mt-4 rounded-2xl border border-amber-200 bg-white px-4 py-3 text-sm font-bold text-amber-700">
          Para cobrar esta cita primero debes abrir caja en la sede correspondiente.
        </div>
      )}
    </div>
  );
}

function AppointmentSaleModal({ branch, cashRegister, appointment, paymentMethods = DEFAULT_PAYMENT_METHODS, labels = readBusinessLabels(), session = null, onClose, onSaved }) {
  const [services, setServices] = useState([]);
  const [barbers, setBarbers] = useState([]);
  const [products, setProducts] = useState([]);
  const [items, setItems] = useState([]);
  const [selectedServiceId, setSelectedServiceId] = useState('');
  const [selectedBarberId, setSelectedBarberId] = useState('');
  const [selectedProductId, setSelectedProductId] = useState('');
  const [quantity, setQuantity] = useState('1');
  const [customerName, setCustomerName] = useState(appointment?.customerName || '');
  const [selectedCustomer, setSelectedCustomer] = useState(null);
  const [customerResults, setCustomerResults] = useState([]);
  const [customerSearching, setCustomerSearching] = useState(false);
  const [paymentMethod, setPaymentMethod] = useState('CASH');
  const [payments, setPayments] = useState([createPaymentDraft('CASH', 0)]);
  const [discount, setDiscount] = useState('0');
  const [tipAmount, setTipAmount] = useState('0');
  const [tipBarberUserId, setTipBarberUserId] = useState('');
  const [cashReceived, setCashReceived] = useState('0');
  const [saleDate, setSaleDate] = useState(toDateInputValue(new Date()));
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const { allowedServiceIds, loadingAllowedServices } = useAllowedServicesForBarber({ barberId: selectedBarberId, branchId: branch?.id, setSelectedServiceId });
  const availableServices = useMemo(() => filterServicesForBarber(services, allowedServiceIds), [services, allowedServiceIds]);

  const paymentOptions = paymentSelectOptions(paymentMethods);

  const subtotal = items.reduce((sum, item) => sum + itemSubtotal(item), 0);
  const discountNumber = Number(String(discount).replace(',', '.')) || 0;

  const appointmentDiscount = appointmentNumber(
    appointment?.discountAmount,
    appointment?.promotionDiscount,
    appointment?.descuento,
    appointment?.descuentoPromocion
  );

  const depositAmount = appointmentNumber(
    appointment?.depositAmount,
    appointment?.montoPagoInicial,
    appointment?.initialPaymentAmount
  );

  const depositStatus = normalizeDepositStatus(
    appointment?.depositStatus ||
      appointment?.estadoPagoInicial ||
      appointment?.depositValidationStatus
  );

  const totalDiscount = roundMoney(discountNumber + appointmentDiscount);
  const totalBeforeDeposit = Math.max(0, roundMoney(subtotal - totalDiscount));
  const depositApplied = isDepositApproved(depositStatus)
    ? Math.min(depositAmount, totalBeforeDeposit)
    : 0;
  const amountToCollectNow = Math.max(0, roundMoney(totalBeforeDeposit - depositApplied));
  const total = totalBeforeDeposit;
  const paymentPayloads = payments
    .map((payment) => ({
      method: normalizeMethod(payment.method),
      amount: roundMoney(parseMoneyInput(payment.amount)),
    }))
    .filter((payment) => payment.method && payment.amount > 0);
  const paymentsTotal = roundMoney(paymentPayloads.reduce((sum, payment) => sum + payment.amount, 0));
  const remainingPayment = roundMoney(amountToCollectNow - paymentsTotal);
  const primaryPaymentMethod = amountToCollectNow <= 0
    ? 'FREE'
    : paymentPayloads.length > 1
      ? 'MIXED'
      : paymentPayloads[0]?.method || paymentMethod || 'CASH';
  const cashPaymentTotal = roundMoney(paymentPayloads
    .filter((payment) => normalizeMethod(payment.method) === 'CASH')
    .reduce((sum, payment) => sum + payment.amount, 0));

  const hasHaircut = items.some((item) => {
    const name = String(item.name || '').toLowerCase();
    return item.type === 'service' && (
      name.includes('corte') ||
      name.includes('fade') ||
      name.includes('taper') ||
      name.includes('degradado') ||
      name.includes('clásico') ||
      name.includes('clasico')
    );
  });

  useEffect(() => {
    async function loadCatalogs() {
      setLoading(true);
      setErrorMsg('');

      try {
        const [serviceData, barberData, productData] = await Promise.all([
          getCashServices(),
          getCashBarbers(branch.id),
          getCashProducts(branch.id).catch(() => []),
        ]);

        const availableBarbers = mergeOwnerIntoBarbers(barberData, session);

        setServices(serviceData);
        setBarbers(availableBarbers);
        setProducts(productData);

        const initialService = serviceData.find(
          (item) => String(item.id) === String(appointment?.serviceId)
        );
        const initialBarber =
          availableBarbers.find((item) => String(item.id) === String(appointment?.barberUserId)) ||
          (String(session?.role || '').trim().toUpperCase() === 'OWNER'
            ? ownerAsBarberOption(session)
            : null);

        if (initialService && initialBarber) {
          setItems([
            {
              key: `appointment-${appointment.appointmentId}`,
              type: 'service',
              serviceId: initialService.id,
              productId: null,
              barberUserId: initialBarber.id,
              barberName: initialBarber.name,
              name: initialService.name,
              quantity: 1,
              unitPrice: servicePriceOf(initialService),
              baseUnitPrice: servicePriceOf(initialService),
              variablePrice: serviceAllowsVariablePrice(initialService),
            },
          ]);
          setSelectedServiceId(String(initialService.id));
          setSelectedBarberId(String(initialBarber.id));
          setCashReceived(String(servicePriceOf(initialService).toFixed(2)));
        }
      } catch (error) {
        setErrorMsg(error.message || 'No se pudieron cargar servicios y ' + labels.professionalsPlural + '.');
      } finally {
        setLoading(false);
      }
    }

    loadCatalogs();
  }, [branch.id, appointment?.appointmentId]);

  useEffect(() => {
    const q = customerName.trim();

    if (selectedCustomer && q === selectedCustomer.nombreCompleto) {
      setCustomerResults([]);
      setCustomerSearching(false);
      return;
    }

    if (q.length < 2) {
      setCustomerResults([]);
      setCustomerSearching(false);
      return;
    }

    let cancelled = false;
    setCustomerSearching(true);

    const timer = window.setTimeout(async () => {
      try {
        const data = await getOwnerCustomers({ query: q, limit: 8 });
        if (cancelled) return;
        setCustomerResults(data);
      } catch {
        if (cancelled) return;
        setCustomerResults([]);
      } finally {
        if (!cancelled) setCustomerSearching(false);
      }
    }, 350);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [customerName, selectedCustomer]);

  function handleCustomerSearchChange(value) {
    setCustomerName(value);

    if (selectedCustomer && value.trim() !== selectedCustomer.nombreCompleto) {
      setSelectedCustomer(null);
    }
  }

  function selectCustomer(customer) {
    setSelectedCustomer(customer);
    setCustomerName(customer.nombreCompleto || customer.nombres || 'Cliente');
    setCustomerResults([]);
    setCustomerSearching(false);
  }

  useEffect(() => {
    setPayments((prev) => {
      if (prev.length !== 1) return prev;
      return [{ ...prev[0], amount: amountToCollectNow.toFixed(2) }];
    });
  }, [amountToCollectNow]);

  useEffect(() => {
    if (cashPaymentTotal > 0 && parseMoneyInput(cashReceived) + 0.009 < cashPaymentTotal) {
      setCashReceived(cashPaymentTotal.toFixed(2));
    }
  }, [cashPaymentTotal, cashReceived]);

  function updatePayment(index, field, value) {
    setPayments((prev) => prev.map((payment, i) => (
      i === index ? { ...payment, [field]: value } : payment
    )));

    if (field === 'method' && index === 0) {
      setPaymentMethod(value);
    }
  }

  function addPaymentMethod() {
    const nextAmount = remainingPayment > 0 ? remainingPayment : 0;
    setPayments((prev) => [...prev, createPaymentDraft(defaultExtraPaymentMethod(paymentMethods), nextAmount)]);
  }

  function removePaymentMethod(index) {
    setPayments((prev) => {
      if (prev.length <= 1) return prev;
      return prev.filter((_, i) => i !== index);
    });
  }

  function addServiceItem() {
    const service = services.find((item) => String(item.id) === String(selectedServiceId));
    const barber = barbers.find((item) => String(item.id) === String(selectedBarberId));
    const qty = Math.max(1, Number(quantity || 1));

    if (!service) {
      setErrorMsg('Selecciona un servicio.');
      return;
    }

    if (!barber) {
      setErrorMsg('Selecciona el ' + labels.professionalSingular + ' que realizó el servicio.');
      return;
    }

    setErrorMsg('');
    setItems((prev) => [
      ...prev,
      {
        key: `service-${service.id}-${barber.id}-${Date.now()}`,
        type: 'service',
        serviceId: service.id,
        productId: null,
        barberUserId: barber.id,
        barberName: barber.name,
        name: service.name,
        quantity: qty,
        unitPrice: servicePriceOf(service),
        baseUnitPrice: servicePriceOf(service),
        variablePrice: serviceAllowsVariablePrice(service),
      },
    ]);
  }

  function addProductItem() {
    const product = products.find((item) => String(item.id) === String(selectedProductId));
    const qty = Math.max(1, Number(quantity || 1));

    if (!product) {
      setErrorMsg('Selecciona un producto.');
      return;
    }

    setErrorMsg('');
    setItems((prev) => [
      ...prev,
      {
        key: `product-${product.id}-${Date.now()}`,
        type: 'product',
        serviceId: null,
        productId: product.id,
        barberUserId: selectedBarberId ? Number(selectedBarberId) : null,
        barberName: barbers.find((item) => String(item.id) === String(selectedBarberId))?.name || null,
        name: product.name,
        quantity: qty,
        unitPrice: Number(product.price || 0),
        baseUnitPrice: Number(product.price || 0),
        priceEditable: true,
      },
    ]);
  }

  function removeItem(key) {
    setItems((prev) => prev.filter((item) => item.key !== key));
  }

  function updateItemUnitPrice(key, value) {
    const nextPrice = Number(String(value).replace(',', '.'));
    setItems((prev) =>
      prev.map((item) =>
        item.key === key
          ? { ...item, unitPrice: Number.isNaN(nextPrice) ? 0 : Math.max(0, nextPrice) }
          : item
      )
    );
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMsg('');

    if (!cashRegister?.id) {
      setErrorMsg('No hay caja abierta para cobrar esta cita.');
      return;
    }

    if (items.length === 0) {
      setErrorMsg('Agrega al menos un servicio o producto.');
      return;
    }

    const received = parseMoneyInput(cashReceived);
    if (amountToCollectNow > 0 && paymentPayloads.length === 0) {
      setErrorMsg('Agrega al menos un metodo de pago.');
      return;
    }

    if (Math.abs(remainingPayment) > 0.009) {
      const label = remainingPayment > 0 ? 'Falta' : 'Sobra';
      setErrorMsg(`${label} ${formatMoney(Math.abs(remainingPayment))} para completar el saldo.`);
      return;
    }

    if (cashPaymentTotal > 0 && received + 0.009 < cashPaymentTotal) {
      setErrorMsg('El efectivo recibido no puede ser menor al monto pagado en efectivo.');
      return;
    }

    setSaving(true);

    try {
      const createdSale = await createCashSale({
        branchId: branch.id,
        customerId: appointment.customerId || null,
        appointmentId: appointment.appointmentId || null,
        saleDate: buildLocalSaleDateForBackend(saleDate),
        metodoPago: primaryPaymentMethod,
        discount: discountNumber,
        cashReceived: cashPaymentTotal > 0 ? received : amountToCollectNow,
        payments:
          amountToCollectNow <= 0
            ? []
            : paymentPayloads,
        cutType: hasHaircut ? 'Corte registrado en agenda web' : null,
        cutDetail: hasHaircut
          ? `${appointment.serviceName || items.find((item) => item.type === 'service')?.name || (labels.serviceReference === 'corte' ? 'Servicio de corte' : 'Servicio')} · ${appointment.barberName || items.find((item) => item.type === 'service')?.barberName || labels.professionalDisplay}`
          : null,
        cutObservations: hasHaircut
          ? `Atención finalizada desde agenda web${appointment.appointmentId ? ` · Cita #${appointment.appointmentId}` : ''}`
          : null,
        items: items.map((item) => ({
          serviceId: item.serviceId,
          productId: item.productId,
          barberUserId: item.barberUserId,
          cantidad: item.quantity,
          precioUnitario: item.unitPrice,
        })),
      });

      await autoPrintApprovedSale(createdSale, { branchId: branch.id });
      clearAttendAppointmentFromStorage();
      offerCustomerWhatsappFollowUp(
        saleWithWhatsappFallback(createdSale, {
          customerName:
            selectedCustomer?.nombreCompleto ||
            customerName.trim() ||
            appointment?.customerName,
          customerPhone:
            selectedCustomer?.telefono ||
            appointment?.customerPhone ||
            appointment?.telefono,
          telefono:
            selectedCustomer?.telefono ||
            appointment?.telefono ||
            appointment?.customerPhone,
        }),
        {
          canOpenWhatsapp: String(session?.role || '').trim().toUpperCase() === 'OWNER',
        }
      );
      onSaved();
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo registrar la venta.');
    } finally {
      setSaving(false);
    }
  }

  const change = cashPaymentTotal > 0
    ? Math.max(0, parseMoneyInput(cashReceived) - cashPaymentTotal)
    : 0;

  return (
    <ModalShell title="Cobrar cita" subtitle={branch?.name || 'Sede'} onClose={onClose}>
      {loading ? (
        <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 text-sm font-black text-neutral-500">
          Cargando servicios y {labels.professionalsPlural}...
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4 pb-1">
          <div className="rounded-[26px] border border-amber-200 bg-amber-50 p-5">
            <div className="text-xs font-black uppercase tracking-[0.20em] text-amber-700">
              Cita seleccionada
            </div>
            <div className="mt-2 text-xl font-black text-neutral-950">
              {appointment.customerName || 'Cliente'} · #{appointment.appointmentId}
            </div>
            <div className="mt-2 text-sm font-bold text-neutral-600">
              Servicio inicial: {appointment.serviceName || 'Servicio'} · {labels.professionalDisplay}: {appointment.barberName || labels.professionalDisplay}
            </div>
          </div>

          <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
            <div className="rounded-[26px] border border-neutral-200 bg-white p-5">
              <div className="text-xs font-black uppercase tracking-[0.20em] text-amber-600">
                Servicios adicionales
              </div>

              <div className="mt-4 grid gap-3 md:grid-cols-2">
                <SelectField
                  label="Servicio"
                  value={selectedServiceId}
                  onChange={setSelectedServiceId}
                  options={[
                    { value: '', label: 'Selecciona servicio' },
                    ...availableServices.map((service) => ({
                      value: String(service.id),
                      label: `${service.name} · ${service.durationMinutes} min · ${servicePriceLabel(service)}`,
                    })),
                  ]}
                />

                <SelectField
                  label={labels.professionalSingular[0].toUpperCase() + labels.professionalSingular.slice(1)}
                  value={selectedBarberId}
                  onChange={setSelectedBarberId}
                  options={[
                    { value: '', label: `Selecciona ${labels.professionalSingular}` },
                    ...barbers.map((barber) => ({
                      value: String(barber.id),
                      label: barber.name,
                    })),
                  ]}
                />
              </div>

              <div className="mt-3 grid gap-3 md:grid-cols-[1fr_160px]">
                <InputField label="Cantidad" value={quantity} onChange={setQuantity} type="number" />
                <button
                  type="button"
                  onClick={addServiceItem}
                  className="self-end rounded-2xl bg-neutral-950 px-5 py-4 text-sm font-black text-white transition hover:scale-[1.01]"
                >
                  Agregar servicio
                </button>
              </div>

              {products.length > 0 && (
                <div className="mt-5 border-t border-neutral-200 pt-5">
                  <div className="text-xs font-black uppercase tracking-[0.20em] text-amber-600">
                    Productos opcionales
                  </div>
                  <div className="mt-3 grid gap-3 md:grid-cols-[1fr_160px]">
                    <SelectField
                      label="Producto"
                      value={selectedProductId}
                      onChange={setSelectedProductId}
                      options={[
                        { value: '', label: 'Selecciona producto' },
                        ...products.map((product) => ({
                          value: String(product.id),
                          label: `${product.name} · ${formatMoney(product.price)} · Stock ${product.stock}`,
                        })),
                      ]}
                    />
                    <button
                      type="button"
                      onClick={addProductItem}
                      className="self-end rounded-2xl border border-neutral-200 bg-white px-5 py-4 text-sm font-black text-neutral-800 transition hover:bg-amber-50"
                    >
                      Agregar producto
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div className="rounded-[26px] border border-neutral-200 bg-white p-5">
              <div className="text-xs font-black uppercase tracking-[0.20em] text-amber-600">
                Pago
              </div>

              <div className="mt-4 space-y-4">
                <label className="block">
                  <span className="text-sm font-black text-neutral-700">Fecha de venta</span>
                  <input
                    type="date"
                    value={saleDate}
                    min={saleDateInputMinValue()}
                    max={saleDateInputMaxValue()}
                    onChange={(event) => setSaleDate(event.target.value)}
                    className="mt-2 w-full rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 font-bold text-neutral-950 outline-none transition placeholder:text-neutral-400 focus:border-amber-400"
                  />
                  {saleDate !== saleDateInputMaxValue() && (
                    <div className="mt-2 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs font-bold text-amber-800">
                      Esta cita se cobrará con fecha anterior para regularizar caja.
                    </div>
                  )}
                </label>                <InputField label="Descuento" value={discount} onChange={setDiscount} type="number" step="0.01" prefix={getTenantCurrencySymbol()} />
                {amountToCollectNow <= 0 ? (
                  <div className="rounded-[24px] border border-emerald-200 bg-emerald-50 p-4">
                    <div className="text-xs font-black uppercase tracking-[0.18em] text-emerald-700">
                      Saldo cubierto
                    </div>
                    <p className="mt-2 text-sm font-bold leading-6 text-emerald-800">
                      No queda monto por cobrar. La venta se completara con el pago inicial aplicado.
                    </p>
                  </div>
                ) : (
                  <div className="rounded-[24px] border border-neutral-200 bg-neutral-50 p-4">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                      <div>
                        <div className="text-xs font-black uppercase tracking-[0.18em] text-amber-600">
                          Metodos de pago
                        </div>
                        <div className="mt-1 text-xs font-bold leading-5 text-neutral-500">
                          Divide el saldo pendiente entre efectivo, Yape, Plin, tarjeta u otros metodos.
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={addPaymentMethod}
                        className="rounded-2xl bg-neutral-950 px-4 py-3 text-xs font-black text-white"
                      >
                        + Agregar metodo
                      </button>
                    </div>

                    <div className="mt-4 space-y-3">
                      {payments.map((payment, index) => (
                        <div key={payment.key} className="rounded-[20px] border border-neutral-200 bg-white p-3">
                          <div className="mb-3 flex items-center justify-between gap-3">
                            <div className="text-xs font-black uppercase tracking-[0.14em] text-neutral-500">
                              {index === 0 ? 'Metodo principal' : `Metodo ${index + 1}`}
                            </div>
                            <button
                              type="button"
                              onClick={() => removePaymentMethod(index)}
                              disabled={payments.length <= 1}
                              className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs font-black text-red-700 disabled:opacity-40"
                            >
                              Quitar
                            </button>
                          </div>

                          <div className="grid gap-3 sm:grid-cols-2">
                            <SelectField
                              label="Metodo"
                              value={payment.method}
                              onChange={(value) => updatePayment(index, 'method', value)}
                              options={paymentOptions}
                            />
                            <InputField
                              label="Monto"
                              value={payment.amount}
                              onChange={(value) => updatePayment(index, 'amount', value)}
                              type="number"
                              step="0.01"
                              prefix={getTenantCurrencySymbol()}
                            />
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className={`mt-4 rounded-2xl border px-4 py-3 text-sm font-black ${
                      Math.abs(remainingPayment) <= 0.009
                        ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                        : remainingPayment > 0
                          ? 'border-amber-200 bg-amber-50 text-amber-700'
                          : 'border-red-200 bg-red-50 text-red-700'
                    }`}>
                      {Math.abs(remainingPayment) <= 0.009
                        ? 'Pago completo'
                        : remainingPayment > 0
                          ? `Falta ${formatMoney(remainingPayment)}`
                          : `Sobra ${formatMoney(Math.abs(remainingPayment))}`}
                    </div>
                  </div>
                )}

                {cashPaymentTotal > 0 && (
                  <InputField label={`Efectivo recibido para ${formatMoney(cashPaymentTotal)}`} value={cashReceived} onChange={setCashReceived} type="number" step="0.01" prefix={getTenantCurrencySymbol()} />
                )}
              </div>
            </div>
          </div>

          <div className="rounded-[26px] border border-neutral-200 bg-white p-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <div className="text-xs font-black uppercase tracking-[0.20em] text-amber-600">
                  Items de venta
                </div>
                <div className="mt-1 text-sm font-bold text-neutral-500">
                  Puedes agregar más servicios o productos antes de cobrar.
                </div>
              </div>
              <div className="rounded-2xl bg-neutral-950 px-4 py-3 text-xl font-black text-white">
                {formatMoney(total)}
              </div>
            </div>

            <div className="mt-4 space-y-3">
              {items.length === 0 ? (
                <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 text-sm font-black text-neutral-500">
                  Aún no hay items en la venta.
                </div>
              ) : (
                items.map((item) => (
                  <div key={item.key} className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <div className="font-black text-neutral-950">{item.name}</div>
                        <div className="mt-1 text-xs font-bold text-neutral-500">
                          {item.type === 'service' ? `Servicio · ${labels.professionalDisplay}: ${item.barberName || '-'}` : `Producto${item.barberName ? ` · ${labels.professionalDisplay}: ${item.barberName}` : ''}`}
                        </div>
                      </div>

                      <div className="flex flex-wrap items-center justify-end gap-3">
                        {(item.type === 'product' || (item.type === 'service' && item.variablePrice)) ? (
                          <label className="min-w-[220px] rounded-2xl border border-amber-200 bg-amber-50 p-3 text-left">
                            <span className="inline-flex rounded-full bg-neutral-950 px-2.5 py-1 text-[10px] font-black uppercase text-white">
                              {item.type === 'product' ? 'Editable' : 'Variable'}
                            </span>
                            <span className="ml-2 text-[11px] font-black text-amber-700">
                              Base {formatMoney(item.baseUnitPrice)}
                            </span>
                            <span className="mt-2 block text-[11px] font-black uppercase tracking-[0.12em] text-neutral-500">
                              Precio final a cobrar
                            </span>
                            <input
                              type="number"
                              min="0"
                              step="0.01"
                              value={item.unitPrice}
                              onChange={(event) => updateItemUnitPrice(item.key, event.target.value)}
                              className="mt-1 h-12 w-full rounded-xl border border-amber-200 bg-white px-3 text-lg font-black text-neutral-950 outline-none focus:border-violet-500"
                            />
                            <span className="mt-1 block text-xs font-bold text-neutral-500">
                              Escribe el monto real de esta venta.
                            </span>
                          </label>
                        ) : null}
                        <div className="text-right">
                          <div className="font-black text-neutral-950">{formatMoney(itemSubtotal(item))}</div>
                          <div className="text-xs font-bold text-neutral-500">
                            {item.variablePrice && item.baseUnitPrice !== item.unitPrice
                              ? `${item.quantity} x ${formatMoney(item.unitPrice)} · desde ${formatMoney(item.baseUnitPrice)}`
                              : `${item.quantity} x ${formatMoney(item.unitPrice)}`}
                          </div>
                        </div>
                      </div>

                      <div className="mt-3">
                        <button
                          type="button"
                          onClick={() => removeItem(item.key)}
                          className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs font-black text-red-700 hover:bg-red-100"
                        >
                          Quitar
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-5">
            <StatCard title="Subtotal" value={formatMoney(subtotal)} />
            <StatCard title="Descuento" value={formatMoney(totalDiscount)} helper={appointmentDiscount > 0 ? `Incluye promo/cita ${formatMoney(appointmentDiscount)}` : null} />
            <StatCard title="Total venta" value={formatMoney(total)} tone="gold" />
            <StatCard title="Inicial aplicado" value={formatMoney(depositApplied)} tone={depositApplied > 0 ? 'green' : 'default'} />
            <StatCard title="Saldo a cobrar" value={formatMoney(amountToCollectNow)} tone="dark" />
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <StatCard title="Vuelto" value={formatMoney(change)} tone={change > 0 ? 'green' : 'default'} />
            <StatCard title="Pago final" value={amountToCollectNow <= 0 ? 'Cubierto con inicial' : paymentLabelFromOptions(paymentMethods, paymentMethod)} helper="El backend completará la venta con DEPOSIT_APPLIED si corresponde." />
          </div>


          <div className="sticky bottom-0 z-30 -mx-5 -mb-4 mt-1 border-t border-neutral-200 bg-white/95 px-5 py-4 shadow-[0_-18px_45px_rgba(15,23,42,0.10)] backdrop-blur-xl sm:-mx-6 sm:-mb-5 sm:px-6">
            {errorMsg && <div className="mb-3"><ErrorBox message={errorMsg} /></div>}
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
              <div className="flex items-center justify-between rounded-2xl bg-neutral-950 px-4 py-3 text-white sm:min-w-[220px]">
                <span className="text-xs font-black uppercase tracking-[0.16em] text-white/50">Total</span>
                <span className="text-xl font-black">{formatMoney(total)}</span>
              </div>
              <button
                disabled={saving || loadingAllowedServices}
                className="min-h-14 flex-1 rounded-2xl bg-amber-400 px-5 py-4 font-black text-neutral-950 shadow-[0_12px_28px_rgba(251,191,36,0.28)] transition hover:bg-amber-300 disabled:opacity-60"
              >
                {saving ? 'Guardando venta...' : 'Cobrar y guardar cita'}
              </button>
            </div>
          </div>
        </form>
      )}
    </ModalShell>
  );
}


function SaleModal({ branch, cashRegister, paymentMethods = DEFAULT_PAYMENT_METHODS, labels = readBusinessLabels(), session = null, onClose, onSaved }) {
  const [services, setServices] = useState([]);
  const [barbers, setBarbers] = useState([]);
  const [products, setProducts] = useState([]);
  const [items, setItems] = useState([]);

  const [selectedServiceId, setSelectedServiceId] = useState('');
  const [selectedProductId, setSelectedProductId] = useState('');
  const [selectedBarberId, setSelectedBarberId] = useState('');
  const [selectedHelperBarberIds, setSelectedHelperBarberIds] = useState([]);
  const [quantity, setQuantity] = useState('1');

  const [customerName, setCustomerName] = useState('');
  const [customerSearchMode, setCustomerSearchMode] = useState('name');
  const [customerSearchPhoneValid, setCustomerSearchPhoneValid] = useState(false);
  const [selectedCustomer, setSelectedCustomer] = useState(null);
  const [customerResults, setCustomerResults] = useState([]);
  const [customerSearching, setCustomerSearching] = useState(false);
  const [quickCustomerFirstName, setQuickCustomerFirstName] = useState('');
  const [quickCustomerPhone, setQuickCustomerPhone] = useState('');
  const [quickCustomerPhoneValid, setQuickCustomerPhoneValid] = useState(false);
  const [quickCustomerLastName, setQuickCustomerLastName] = useState('');
  const [creatingCustomer, setCreatingCustomer] = useState(false);
  const [isCourtesy, setIsCourtesy] = useState(false);
  const [paymentMethod, setPaymentMethod] = useState('CASH');
  const [discount, setDiscount] = useState('0');
  const [tipAmount, setTipAmount] = useState('0');
  const [tipBarberUserId, setTipBarberUserId] = useState('');
  const [cashReceived, setCashReceived] = useState('0');
  const [saleDate, setSaleDate] = useState(toDateInputValue(new Date()));
  const [payments, setPayments] = useState([createPaymentDraft('CASH', 0)]);
  const [documentType, setDocumentType] = useState('NONE');
  const [electronicInvoicingAvailable, setElectronicInvoicingAvailable] = useState(false);
  const [receiverDocumentNumber, setReceiverDocumentNumber] = useState('');
  const [receiverName, setReceiverName] = useState('');
  const [receiverAddress, setReceiverAddress] = useState('');
  const [receiverEmail, setReceiverEmail] = useState('');
  const [printElectronicDocument, setPrintElectronicDocument] = useState(false);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const { allowedServiceIds, loadingAllowedServices } = useAllowedServicesForBarber({ barberId: selectedBarberId, branchId: branch?.id, setSelectedServiceId });
  const availableServices = useMemo(() => filterServicesForBarber(services, allowedServiceIds), [services, allowedServiceIds]);

  const paymentOptions = paymentSelectOptions(paymentMethods);

  const subtotal = items.reduce((sum, item) => sum + itemSubtotal(item), 0);
  const discountNumber = Number(String(discount).replace(',', '.')) || 0;
  const tipNumber = Math.max(0, parseMoneyInput(tipAmount));
  const grossTotal = roundMoney(subtotal + tipNumber);
  const effectiveDiscount = isCourtesy ? grossTotal : discountNumber;
  const firstServiceBarberId = items.find((item) => item.type === 'service' && item.barberUserId)?.barberUserId || selectedBarberId || '';
  const effectiveTipBarberUserId = tipNumber > 0
    ? (Number(tipBarberUserId || firstServiceBarberId || 0) || null)
    : null;
  const total = Math.max(0, roundMoney(grossTotal - effectiveDiscount));
  const paymentPayloads = payments
    .map((payment) => ({
      method: normalizeMethod(payment.method),
      amount: roundMoney(parseMoneyInput(payment.amount)),
    }))
    .filter((payment) => payment.method && payment.amount > 0);
  const paymentsTotal = roundMoney(paymentPayloads.reduce((sum, payment) => sum + payment.amount, 0));
  const remainingPayment = roundMoney(total - paymentsTotal);
  const hasMixedPayment = paymentPayloads.length > 1;
  const primaryPaymentMethod = total === 0
    ? 'FREE'
    : hasMixedPayment
      ? 'MIXED'
      : paymentPayloads[0]?.method || paymentMethod || 'CASH';
  const cashPaymentTotal = roundMoney(paymentPayloads
    .filter((payment) => normalizeMethod(payment.method) === 'CASH')
    .reduce((sum, payment) => sum + payment.amount, 0));
  const received = parseMoneyInput(cashReceived);
  const change = cashPaymentTotal > 0 ? Math.max(0, roundMoney(received - cashPaymentTotal)) : 0;

  useEffect(() => {
    async function loadCatalogs() {
      setLoading(true);
      setErrorMsg('');

      try {
        const [serviceData, barberData, productData, invoicingAccess] = await Promise.all([
          getCashServices(),
          getCashBarbers(branch.id),
          getCashProducts(branch.id).catch(() => []),
          getElectronicInvoicingAccess().catch(() => ({ available: false })),
        ]);

        setServices(serviceData);
        setBarbers(mergeOwnerIntoBarbers(barberData, session));
        setProducts(productData);
        setElectronicInvoicingAvailable(invoicingAccess?.available === true);
      } catch (error) {
        setErrorMsg(error.message || `No se pudieron cargar servicios, productos y ${labels.professionalsPlural}.`);
      } finally {
        setLoading(false);
      }
    }

    loadCatalogs();
  }, [branch.id, session]);

  useEffect(() => {
    const q = customerName.trim();

    if (customerSearchMode === 'phone' && !customerSearchPhoneValid) {
      setCustomerResults([]);
      setCustomerSearching(false);
      return;
    }

    if (selectedCustomer && q === (selectedCustomer.nombreCompleto || selectedCustomer.nombres || 'Cliente')) {
      setCustomerResults([]);
      setCustomerSearching(false);
      return;
    }

    if (q.length < 2) {
      setCustomerResults([]);
      setCustomerSearching(false);
      return;
    }

    let cancelled = false;
    setCustomerSearching(true);

    const timer = window.setTimeout(async () => {
      try {
        const data = await getOwnerCustomers({ query: q, limit: 8 });
        if (cancelled) return;
        setCustomerResults(Array.isArray(data) ? data : []);
      } catch {
        if (cancelled) return;
        setCustomerResults([]);
      } finally {
        if (!cancelled) setCustomerSearching(false);
      }
    }, 350);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [customerName, selectedCustomer, customerSearchMode, customerSearchPhoneValid]);

  function handleCustomerSearchChange(value) {
    setCustomerName(value);

    if (selectedCustomer) {
      const selectedName = selectedCustomer.nombreCompleto || selectedCustomer.nombres || 'Cliente';
      if (value.trim() !== selectedName) {
        setSelectedCustomer(null);
      }
    }

    if (value.trim() && !selectedCustomer) {
      setQuickCustomerFirstName(value.trim());
    }
  }

  function selectCustomer(customer) {
    setSelectedCustomer(customer);
    setCustomerName(customer.nombreCompleto || customer.nombres || 'Cliente');
    setCustomerSearchMode('name');
    setCustomerSearchPhoneValid(false);
    setQuickCustomerFirstName('');
    setQuickCustomerPhone(customer.telefono || '');
    setQuickCustomerLastName('');
    setCustomerResults([]);
    setCustomerSearching(false);
    setReceiverName(customer.nombreCompleto || [customer.nombres, customer.apellidos].filter(Boolean).join(' ') || 'Cliente');
    if (customer.email) setReceiverEmail(customer.email);
  }

  async function createQuickCustomerFromSale() {
    setErrorMsg('');

    const name = (quickCustomerFirstName.trim() || (customerSearchMode === 'name' ? customerName.trim() : '')).trim();
    const lastName = quickCustomerLastName.trim();
    const phone = quickCustomerPhone;

    if (!name) {
      setErrorMsg('Escribe el nombre del cliente para crearlo.');
      return;
    }

    if (!quickCustomerPhoneValid || !phone) {
      setErrorMsg('Ingresa un teléfono válido para crear el cliente.');
      return;
    }

    setCreatingCustomer(true);

    try {
      const created = await createOwnerCustomer({
        nombres: name,
        apellidos: lastName || null,
        telefono: phone,
      });
      selectCustomer(created);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo crear el cliente.');
    } finally {
      setCreatingCustomer(false);
    }
  }

  useEffect(() => {
    setPayments((prev) => {
      if (prev.length !== 1) return prev;
      return [{ ...prev[0], amount: total.toFixed(2) }];
    });
  }, [total]);

  useEffect(() => {
    if (cashPaymentTotal > 0 && received + 0.009 < cashPaymentTotal) {
      setCashReceived(cashPaymentTotal.toFixed(2));
    }
  }, [cashPaymentTotal]);

  function updatePayment(index, field, value) {
    setPayments((prev) => prev.map((payment, i) => (
      i === index ? { ...payment, [field]: value } : payment
    )));

    if (field === 'method' && index === 0) {
      setPaymentMethod(value);
    }
  }

  function addPaymentMethod() {
    const nextAmount = remainingPayment > 0 ? remainingPayment : 0;
    setPayments((prev) => [...prev, createPaymentDraft(defaultExtraPaymentMethod(paymentMethods), nextAmount)]);
  }

  function removePaymentMethod(index) {
    setPayments((prev) => {
      if (prev.length <= 1) return prev;
      return prev.filter((_, i) => i !== index);
    });
  }

  function currentBarber() {
    return barbers.find((item) => String(item.id) === String(selectedBarberId));
  }

  function toggleHelperBarber(barberId) {
    const id = String(barberId);
    setSelectedHelperBarberIds((prev) =>
      prev.includes(id)
        ? prev.filter((item) => item !== id)
        : [...prev, id]
    );
  }

  function addServiceItem() {
    const service = services.find((item) => String(item.id) === String(selectedServiceId));
    const barber = currentBarber();
    const qty = Math.max(1, Number(quantity || 1));

    if (!service) {
      setErrorMsg('Selecciona un servicio.');
      return;
    }

    if (!barber) {
      setErrorMsg(`Selecciona el ${labels.professionalSingular} que realizo el servicio.`);
      return;
    }

    const helperBarbers = selectedHelperBarberIds
      .filter((id) => id !== String(barber.id))
      .map((id) => barbers.find((item) => String(item.id) === id))
      .filter(Boolean);
    const assignedBarbers = [barber, ...helperBarbers];
    const basePrice = servicePriceOf(service);
    const totalServiceAmount = roundMoney(basePrice * qty);
    const splitAmount = assignedBarbers.length > 1
      ? roundMoney(totalServiceAmount / assignedBarbers.length)
      : basePrice;

    setErrorMsg('');
    setItems((prev) => [
      ...prev,
      ...assignedBarbers.map((assignedBarber, index) => ({
        key: `service-${service.id}-${assignedBarber.id}-${Date.now()}-${index}`,
        type: 'service',
        serviceId: service.id,
        productId: null,
        barberUserId: assignedBarber.id,
        barberName: assignedBarber.name,
        name: service.name,
        quantity: assignedBarbers.length > 1 ? 1 : qty,
        unitPrice: splitAmount,
        baseUnitPrice: assignedBarbers.length > 1 ? splitAmount : basePrice,
        variablePrice: serviceAllowsVariablePrice(service),
      })),
    ]);

    setSelectedServiceId('');
    setSelectedHelperBarberIds([]);
    setQuantity('1');
  }

  function addProductItem() {
    const product = products.find((item) => String(item.id) === String(selectedProductId));
    const barber = currentBarber();
    const qty = Math.max(1, Number(quantity || 1));

    if (!product) {
      setErrorMsg('Selecciona un producto.');
      return;
    }

    const stock = Number(product.stock ?? product.stockActual ?? 0);
    const allowsWithoutStock = product.permiteVentaSinStock === true || product.allowSaleWithoutStock === true;

    if (!allowsWithoutStock && stock < qty) {
      setErrorMsg(`Stock insuficiente. Disponible: ${stock}.`);
      return;
    }

    const commission = Number(product.barberCommissionAmount ?? product.productCommissionAmount ?? 0);
    if (commission > 0 && !barber) {
      setErrorMsg(`Este producto tiene comision. Selecciona el ${labels.professionalSingular} que hizo la venta.`);
      return;
    }

    setErrorMsg('');
    setItems((prev) => [
      ...prev,
      {
        key: `product-${product.id}-${Date.now()}`,
        type: 'product',
        serviceId: null,
        productId: product.id,
        barberUserId: barber?.id || null,
        barberName: barber?.name || null,
        name: product.name,
        quantity: qty,
        unitPrice: Number(product.price ?? product.precioVenta ?? 0),
        baseUnitPrice: Number(product.price ?? product.precioVenta ?? 0),
        priceEditable: true,
      },
    ]);

    setSelectedProductId('');
    setQuantity('1');
  }

  function removeItem(key) {
    setItems((prev) => prev.filter((item) => item.key !== key));
  }

  function updateItemUnitPrice(key, value) {
    const nextPrice = Number(String(value).replace(',', '.'));
    setItems((prev) =>
      prev.map((item) =>
        item.key === key
          ? { ...item, unitPrice: Number.isNaN(nextPrice) ? 0 : Math.max(0, nextPrice) }
          : item
      )
    );
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMsg('');

    if (!cashRegister?.id) {
      setErrorMsg('No hay caja abierta para registrar la venta.');
      return;
    }

    if (items.length === 0) {
      setErrorMsg('Agrega al menos un servicio o producto.');
      return;
    }

    if (tipNumber > 0 && !effectiveTipBarberUserId) {
      setErrorMsg(`Para registrar propina, selecciona el ${labels.professionalSingular} que la recibira o agrega un servicio con ${labels.professionalSingular}.`);
      return;
    }

    if (!isCourtesy && total > 0 && paymentPayloads.length === 0) {
      setErrorMsg('Agrega al menos un método de pago.');
      return;
    }

    if (!isCourtesy && Math.abs(remainingPayment) > 0.009) {
      const label = remainingPayment > 0 ? 'Falta' : 'Sobra';
      setErrorMsg(`${label} ${formatMoney(Math.abs(remainingPayment))} para completar el total.`);
      return;
    }

    if (!isCourtesy && cashPaymentTotal > 0 && received + 0.009 < cashPaymentTotal) {
      setErrorMsg('El efectivo recibido no puede ser menor al monto pagado en efectivo.');
      return;
    }

    if (documentType !== 'NONE') {
      const digits = receiverDocumentNumber.replace(/\D/g, '');
      if (total <= 0) { setErrorMsg('No se puede emitir comprobante electrónico para una venta gratuita.'); return; }
      if (!receiverName.trim()) { setErrorMsg('Escribe el nombre o razón social del receptor.'); return; }
      if (documentType === 'RECEIPT' && !/^\d{8}$/.test(digits)) {
        setErrorMsg('La boleta de esta primera versión requiere un DNI válido de 8 dígitos.'); return;
      }
      if (documentType === 'INVOICE' && !/^\d{11}$/.test(digits)) {
        setErrorMsg('La factura requiere un RUC válido de 11 dígitos.'); return;
      }
      if (documentType === 'INVOICE' && !receiverAddress.trim()) {
        setErrorMsg('La dirección fiscal es obligatoria para la factura.'); return;
      }
    }

    const hasHaircut = items.some((item) => {
      const name = String(item.name || '').toLowerCase();
      return item.type === 'service' && (
        name.includes('corte') ||
        name.includes('fade') ||
        name.includes('taper') ||
        name.includes('degradado') ||
        name.includes('clásico') ||
        name.includes('clasico')
      );
    });

    const preparedPrintPopup = documentType !== 'NONE' && printElectronicDocument
      ? window.open('', '_blank')
      : null;
    if (preparedPrintPopup) {
      preparedPrintPopup.document.write('<title>Preparando comprobante</title><p style="font-family:system-ui;padding:32px">Guardando la venta y preparando el comprobante oficial...</p>');
    }

    setSaving(true);

    try {
      const createdSale = await createCashSale({
        branchId: branch.id,
        customerId: selectedCustomer?.id || null,
        appointmentId: null,
        saleDate: buildLocalSaleDateForBackend(saleDate),
        metodoPago: primaryPaymentMethod,
        discount: effectiveDiscount,
        cashReceived: isCourtesy ? 0 : cashPaymentTotal > 0 ? received : total,
        tipAmount: tipNumber,
        tipBarberUserId: effectiveTipBarberUserId,
        payments: isCourtesy || total === 0 ? [] : paymentPayloads,
        cutType: hasHaircut ? 'Corte registrado en caja web' : null,
        cutDetail: hasHaircut ? 'Venta registrada desde panel web' : null,
        cutObservations: selectedCustomer ? null : (customerName.trim() ? `Referencia: ${customerName.trim()}` : null),
        items: items.map((item) => ({
          serviceId: item.serviceId,
          productId: item.productId,
          barberUserId: item.barberUserId,
          cantidad: item.quantity,
          precioUnitario: item.unitPrice,
        })),
      });

      let electronicDocument = null;
      let electronicDocumentError = null;
      if (documentType !== 'NONE') {
        try {
          electronicDocument = await issueElectronicDocument({
            saleId: saleIdOf(createdSale), branchId: branch.id, documentType,
            receiverDocumentType: documentType === 'INVOICE' ? '6' : '1',
            receiverDocumentNumber: receiverDocumentNumber.replace(/\D/g, ''),
            receiverName: receiverName.trim(), receiverAddress: receiverAddress.trim() || '-',
            receiverEmail: receiverEmail.trim() || null,
          });
        } catch (reason) {
          electronicDocumentError = reason.message || 'No se pudo emitir el comprobante.';
        }
      }

      if (documentType === 'NONE' && printElectronicDocument) {
        await autoPrintApprovedSale(createdSale, { branchId: branch.id });
      }
      offerCustomerWhatsappFollowUp(
        saleWithWhatsappFallback(createdSale, {
          customerName:
            selectedCustomer?.nombreCompleto ||
            customerName.trim() ||
            'Cliente',
          customerPhone: selectedCustomer?.telefono || quickCustomerPhone,
          telefono: selectedCustomer?.telefono || quickCustomerPhone,
        }),
        {
          canOpenWhatsapp: String(session?.role || '').trim().toUpperCase() === 'OWNER',
        }
      );
      if (electronicDocument) {
        const number = [electronicDocument.series, electronicDocument.sequence].filter(Boolean).join('-');
        const accepted = electronicDocument.status === 'ACCEPTED' || electronicDocument.status === 'ACCEPTED_WITH_OBSERVATIONS';
        if (accepted) {
          if (printElectronicDocument) {
            await openElectronicPdfForPrint(electronicDocument, preparedPrintPopup);
          } else {
            window.alert(`Venta guardada. ${documentType === 'INVOICE' ? 'Factura' : 'Boleta'} ${number} aceptada por SUNAT.`);
          }
        } else {
          preparedPrintPopup?.close();
          window.alert(`Venta guardada. El comprobante ${number || 'sin número'} quedó en estado ${electronicDocument.status}. Puedes actualizarlo desde Ver detalle.`);
        }
      } else if (electronicDocumentError) {
        preparedPrintPopup?.close();
        window.alert(`La venta fue guardada correctamente, pero el comprobante quedó pendiente: ${electronicDocumentError}`);
      }
      onSaved(createdSale, {
        electronicDocumentError,
        requestedDocumentType: documentType,
        pendingElectronicDocument: electronicDocumentError ? {
          documentType,
          receiverDocumentNumber: receiverDocumentNumber.replace(/\D/g, ''),
          receiverName: receiverName.trim(),
          receiverAddress: receiverAddress.trim(),
          receiverEmail: receiverEmail.trim(),
          printAfterIssue: printElectronicDocument,
        } : null,
      });
    } catch (error) {
      preparedPrintPopup?.close();
      setErrorMsg(error.message || 'No se pudo registrar la venta.');
    } finally {
      setSaving(false);
    }
  }

  return (
    <ModalShell title="Nueva venta" subtitle={branch?.name || 'Sede'} onClose={onClose}>
      {loading ? (
        <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 text-sm font-black text-neutral-500">
          Cargando servicios, productos y {labels.professionalsPlural}...
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4 pb-1">
          <div className="grid items-start gap-4 xl:grid-cols-[1.15fr_0.85fr]">
            <div className="rounded-[26px] border border-neutral-200 bg-white p-5">
              <div className="text-xs font-black uppercase tracking-[0.20em] text-amber-600">
                Venta rápida
              </div>
              <h3 className="mt-2 text-2xl font-black text-neutral-950">
                Servicios y productos
              </h3>
              <p className="mt-1 text-sm font-semibold text-neutral-500">
                Agrega servicios o productos. El stock se descontará de la sede activa.
              </p>

              <div className="mt-5 grid gap-3 md:grid-cols-2">
                <div className="relative">
                  <div className="mb-2 grid grid-cols-2 gap-2 rounded-2xl bg-neutral-100 p-1.5">
                    {[
                      ['name', 'Por nombre'],
                      ['phone', 'Por WhatsApp'],
                    ].map(([mode, label]) => (
                      <button
                        key={mode}
                        type="button"
                        onClick={() => {
                          setCustomerSearchMode(mode);
                          setCustomerName('');
                          setCustomerSearchPhoneValid(false);
                          setSelectedCustomer(null);
                          setCustomerResults([]);
                        }}
                        className={`rounded-xl px-3 py-2 text-xs font-black transition ${
                          customerSearchMode === mode
                            ? 'bg-neutral-950 text-white'
                            : 'text-neutral-500 hover:bg-white'
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>

                  {customerSearchMode === 'phone' ? (
                    <InternationalPhoneField
                      label="WhatsApp para buscar"
                      value={customerName}
                      compact
                      onChange={(e164, meta) => {
                        setCustomerName(e164);
                        setCustomerSearchPhoneValid(meta.isValid);
                        if (meta.isValid) {
                          setQuickCustomerPhone(e164);
                          setQuickCustomerPhoneValid(true);
                        }
                      }}
                      helperText={customerName && !customerSearchPhoneValid
                        ? 'Completa el numero para buscar.'
                        : 'Selecciona el pais correcto del cliente.'}
                    />
                  ) : (
                    <label className="block">
                      <span className="text-sm font-black text-neutral-700">Cliente</span>
                      <input
                        value={customerName}
                        onChange={(event) => handleCustomerSearchChange(event.target.value)}
                        placeholder="Nombre o apellido"
                        className="mt-2 w-full rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 font-bold text-neutral-950 outline-none transition placeholder:text-neutral-400 focus:border-amber-400"
                      />
                    </label>
                  )}

                  {customerSearching && (
                    <div className="mt-2 rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm font-black text-neutral-500">
                      Buscando clientes...
                    </div>
                  )}

                  {selectedCustomer && (
                    <div className="mt-2 flex items-center justify-between gap-3 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3">
                      <div>
                        <div className="text-sm font-black text-emerald-800">
                          {selectedCustomer.nombreCompleto}
                        </div>
                        <div className="mt-1 text-xs font-bold text-emerald-700/75">
                          {selectedCustomer.telefono || 'Sin teléfono'} · {selectedCustomer.puntosDisponibles || 0} pts
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          setSelectedCustomer(null);
                          setCustomerSearchMode('name');
                          setCustomerSearchPhoneValid(false);
                          setCustomerName('');
                          setQuickCustomerFirstName('');
                          setQuickCustomerPhone('');
                          setQuickCustomerLastName('');
                          setCustomerResults([]);
                        }}
                        className="rounded-xl bg-white px-3 py-2 text-xs font-black text-emerald-700"
                      >
                        Cambiar
                      </button>
                    </div>
                  )}

                  {!selectedCustomer && customerResults.length > 0 && (
                    <div className="absolute left-0 right-0 z-20 mt-2 max-h-72 overflow-auto rounded-2xl border border-neutral-200 bg-white p-2 shadow-[0_20px_60px_rgba(15,23,42,0.18)]">
                      {customerResults.map((customer) => (
                        <button
                          key={customer.id}
                          type="button"
                          onClick={() => selectCustomer(customer)}
                          className="flex w-full items-center justify-between gap-3 rounded-xl px-3 py-3 text-left transition hover:bg-amber-50"
                        >
                          <div>
                            <div className="font-black text-neutral-950">
                              {customer.nombreCompleto}
                            </div>
                            <div className="mt-1 text-xs font-bold text-neutral-500">
                              {customer.telefono || 'Sin teléfono'} · {customer.puntosDisponibles || 0} pts
                            </div>
                          </div>
                          <span className="rounded-full bg-neutral-950 px-3 py-1 text-xs font-black text-white">
                            Elegir
                          </span>
                        </button>
                      ))}
                    </div>
                  )}

                  {!selectedCustomer && !customerSearching && customerName.trim().length >= 2 && customerResults.length === 0 && (customerSearchMode !== 'phone' || customerSearchPhoneValid) && (
                    <div className="mt-2 rounded-2xl border border-amber-200 bg-amber-50 p-4">
                      <div className="text-xs font-black uppercase tracking-[0.16em] text-amber-700">
                        Cliente no encontrado
                      </div>
                      <p className="mt-2 text-xs font-bold leading-5 text-amber-800">
                        Puedes guardar la venta como cliente ocasional o crear el cliente ahora agregando su teléfono.
                      </p>

                      <div className="mt-3 grid gap-3 sm:grid-cols-2">
                        <input
                          value={quickCustomerFirstName}
                          onChange={(event) => setQuickCustomerFirstName(event.target.value)}
                          placeholder="Nombre del cliente"
                          className="w-full rounded-2xl border border-amber-200 bg-white px-4 py-3 text-sm font-black text-neutral-950 outline-none transition placeholder:text-neutral-400 focus:border-amber-500"
                        />

                        <input
                          value={quickCustomerLastName}
                          onChange={(event) => setQuickCustomerLastName(event.target.value)}
                          placeholder="Apellido del cliente"
                          className="w-full rounded-2xl border border-amber-200 bg-white px-4 py-3 text-sm font-black text-neutral-950 outline-none transition placeholder:text-neutral-400 focus:border-amber-500"
                        />
                        <InternationalPhoneField
                          label="WhatsApp del cliente"
                          value={quickCustomerPhone}
                          compact
                          className="sm:col-span-2"
                          onChange={(e164, meta) => {
                            setQuickCustomerPhone(e164);
                            setQuickCustomerPhoneValid(meta.isValid);
                          }}
                        />

                        <button
                          type="button"
                          onClick={createQuickCustomerFromSale}
                          disabled={creatingCustomer}
                          className="w-full rounded-2xl bg-neutral-950 px-4 py-3 text-sm font-black text-white transition hover:scale-[1.01] disabled:opacity-60 sm:w-auto"
                        >
                          {creatingCustomer ? 'Creando...' : 'Crear cliente'}
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                <SelectField
                  label={labels.professionalDisplay}
                  value={selectedBarberId}
                  onChange={(value) => {
                    setSelectedBarberId(value);
                    setSelectedHelperBarberIds((prev) => prev.filter((id) => id !== String(value)));
                  }}
                  options={[
                    { value: '', label: 'Selecciona ' + labels.professionalSingular },
                    ...barbers.map((barber) => ({
                      value: String(barber.id),
                      label: barber.name,
                    })),
                  ]}
                />
              </div>

              {selectedBarberId && barbers.length > 1 && (
                <div className="mt-3 rounded-[22px] border border-amber-200 bg-amber-50 p-4">
                  <div className="text-xs font-black uppercase tracking-[0.16em] text-amber-700">
                    Servicio compartido
                  </div>
                  <p className="mt-1 text-xs font-bold leading-5 text-amber-800">
                    Marca otros profesionales si este servicio se realizo entre varias personas. El monto se divide para calcular comisiones.
                  </p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {barbers
                      .filter((barber) => String(barber.id) !== String(selectedBarberId))
                      .map((barber) => {
                        const selected = selectedHelperBarberIds.includes(String(barber.id));
                        return (
                          <button
                            key={barber.id}
                            type="button"
                            onClick={() => toggleHelperBarber(barber.id)}
                            className={`rounded-full border px-3 py-2 text-xs font-black transition ${
                              selected
                                ? 'border-neutral-950 bg-neutral-950 text-white'
                                : 'border-amber-200 bg-white text-amber-800 hover:bg-amber-100'
                            }`}
                          >
                            {barber.name}
                          </button>
                        );
                      })}
                  </div>
                </div>
              )}

              <div className="mt-5 rounded-[24px] border border-neutral-200 bg-neutral-50 p-4">
                <div className="text-sm font-black text-neutral-950">Agregar servicio</div>
                <div className="mt-3 grid gap-3 md:grid-cols-[1fr_120px_170px]">
                  <SelectField
                    label="Servicio"
                    value={selectedServiceId}
                    onChange={setSelectedServiceId}
                    options={[
                      { value: '', label: 'Selecciona servicio' },
                      ...availableServices.map((service) => ({
                        value: String(service.id),
                        label: `${service.name} · ${service.durationMinutes} min · ${servicePriceLabel(service)}`,
                      })),
                    ]}
                  />

                  <InputField label="Cantidad" value={quantity} onChange={setQuantity} type="number" />

                  <button
                    type="button"
                    onClick={addServiceItem}
                    className="self-end rounded-2xl bg-neutral-950 px-5 py-4 text-sm font-black text-white transition hover:scale-[1.01]"
                  >
                    Agregar servicio
                  </button>
                </div>
              </div>

              <div className="mt-4 rounded-[24px] border border-emerald-200 bg-emerald-50 p-4">
                <div className="text-sm font-black text-neutral-950">Agregar producto</div>
                <div className="mt-3 grid gap-3 md:grid-cols-[1fr_120px_170px]">
                  <SelectField
                    label="Producto"
                    value={selectedProductId}
                    onChange={setSelectedProductId}
                    options={[
                      { value: '', label: 'Selecciona producto' },
                      ...products.map((product) => ({
                        value: String(product.id),
                        label: `${product.name} · ${formatMoney(product.price)} · Stock ${product.stock}`,
                      })),
                    ]}
                  />

                  <InputField label="Cantidad" value={quantity} onChange={setQuantity} type="number" />

                  <button
                    type="button"
                    onClick={addProductItem}
                    className="self-end rounded-2xl border border-emerald-200 bg-white px-5 py-4 text-sm font-black text-emerald-700 transition hover:bg-emerald-100"
                  >
                    Agregar producto
                  </button>
                </div>
              </div>

          <div className="rounded-[26px] border border-neutral-200 bg-white p-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <div className="text-xs font-black uppercase tracking-[0.20em] text-amber-600">
                  Items de venta
                </div>
                <div className="mt-1 text-sm font-bold text-neutral-500">
                  {items.length} item(s) agregado(s)
                </div>
              </div>
              <div className="rounded-2xl bg-neutral-950 px-4 py-3 text-xl font-black text-white">
                {formatMoney(total)}
              </div>
            </div>

            <div className="mt-4 space-y-3">
              {items.length === 0 ? (
                <div className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 text-sm font-black text-neutral-500">
                  Aún no hay items en la venta.
                </div>
              ) : (
                items.map((item) => (
                  <div key={item.key} className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <div className="font-black text-neutral-950">{item.name}</div>
                        <div className="mt-1 text-xs font-bold text-neutral-500">
                          {item.type === 'service'
                            ? `Servicio · ${labels.professionalDisplay}: ${item.barberName || '-'}`
                            : `Producto${item.barberName ? ` · ${labels.professionalDisplay}: ${item.barberName}` : ''}`}
                        </div>
                      </div>

                      <div className="flex flex-wrap items-center justify-end gap-3">
                        {(item.type === 'product' || (item.type === 'service' && item.variablePrice)) ? (
                          <label className="min-w-[220px] rounded-2xl border border-amber-200 bg-amber-50 p-3 text-left">
                            <span className="inline-flex rounded-full bg-neutral-950 px-2.5 py-1 text-[10px] font-black uppercase text-white">
                              {item.type === 'product' ? 'Editable' : 'Variable'}
                            </span>
                            <span className="ml-2 text-[11px] font-black text-amber-700">
                              Base {formatMoney(item.baseUnitPrice)}
                            </span>
                            <span className="mt-2 block text-[11px] font-black uppercase tracking-[0.12em] text-neutral-500">
                              Precio final a cobrar
                            </span>
                            <input
                              type="number"
                              min="0"
                              step="0.01"
                              value={item.unitPrice}
                              onChange={(event) => updateItemUnitPrice(item.key, event.target.value)}
                              className="mt-1 h-12 w-full rounded-xl border border-amber-200 bg-white px-3 text-lg font-black text-neutral-950 outline-none focus:border-violet-500"
                            />
                            <span className="mt-1 block text-xs font-bold text-neutral-500">
                              Escribe el monto real de esta venta.
                            </span>
                          </label>
                        ) : null}
                        <div className="text-right">
                          <div className="font-black text-neutral-950">{formatMoney(itemSubtotal(item))}</div>
                          <div className="text-xs font-bold text-neutral-500">
                            {item.variablePrice && item.baseUnitPrice !== item.unitPrice
                              ? `${item.quantity} x ${formatMoney(item.unitPrice)} · desde ${formatMoney(item.baseUnitPrice)}`
                              : `${item.quantity} x ${formatMoney(item.unitPrice)}`}
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => removeItem(item.key)}
                          className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs font-black text-red-700 hover:bg-red-100"
                        >
                          Quitar
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

            </div>

            <div className="rounded-[26px] border border-neutral-200 bg-white p-5">
              <div className="text-xs font-black uppercase tracking-[0.20em] text-amber-600">
                Cobro
              </div>
              <h3 className="mt-2 text-2xl font-black text-neutral-950">
                {formatMoney(total)}
              </h3>
              <p className="mt-1 text-sm font-semibold text-neutral-500">
                Total a cobrar
              </p>

              <button
                type="button"
                onClick={() => setIsCourtesy((value) => !value)}
                className={`mt-4 flex w-full items-center justify-between gap-4 rounded-[22px] border px-4 py-4 text-left transition ${
                  isCourtesy
                    ? 'border-amber-300 bg-amber-50'
                    : 'border-neutral-200 bg-neutral-50 hover:border-amber-200 hover:bg-amber-50/50'
                }`}
              >
                <span>
                  <span className="block text-sm font-black text-neutral-950">
                    Registrar como cortesia gratis
                  </span>
                  <span className="mt-1 block text-xs font-bold leading-5 text-neutral-500">
                    Guarda la venta en 0 y conserva el subtotal como referencia para reportes.
                  </span>
                </span>
                <span className={`flex h-7 w-12 shrink-0 items-center rounded-full p-1 transition ${
                  isCourtesy ? 'bg-amber-400' : 'bg-neutral-300'
                }`}>
                  <span className={`h-5 w-5 rounded-full bg-white shadow-sm transition ${
                    isCourtesy ? 'translate-x-5' : 'translate-x-0'
                  }`} />
                </span>
              </button>

              <div className="mt-5 space-y-4">
                {electronicInvoicingAvailable && (
                  <div className="rounded-[24px] border border-violet-200 bg-violet-50 p-4">
                    <div className="text-xs font-black uppercase tracking-[0.18em] text-violet-700">Comprobante electrónico</div>
                    <p className="mt-1 text-xs font-bold text-violet-700/70">Emite mediante Mifact y SUNAT.</p>
                    <div className="mt-3 grid grid-cols-3 gap-2">
                      {[['NONE','Sin comprobante'],['RECEIPT','Boleta'],['INVOICE','Factura']].map(([value,label]) => (
                        <button key={value} type="button" onClick={() => { setDocumentType(value); setPrintElectronicDocument(value !== 'NONE'); }}
                          className={`rounded-xl px-2 py-3 text-xs font-black ${documentType === value ? 'bg-neutral-950 text-white' : 'border border-violet-200 bg-white text-neutral-600'}`}>{label}</button>
                      ))}
                    </div>
                    {documentType !== 'NONE' && <div className="mt-4 space-y-3">
                      <InputField label={documentType === 'INVOICE' ? 'RUC (11 dígitos)' : 'DNI (8 dígitos)'} value={receiverDocumentNumber} onChange={(value) => setReceiverDocumentNumber(value.replace(/\D/g, '').slice(0, documentType === 'INVOICE' ? 11 : 8))} />
                      <InputField label={documentType === 'INVOICE' ? 'Razón social' : 'Nombre del cliente'} value={receiverName} onChange={setReceiverName} />
                      <InputField label={documentType === 'INVOICE' ? 'Dirección fiscal' : 'Dirección (opcional)'} value={receiverAddress} onChange={setReceiverAddress} />
                      <InputField label="Correo opcional" value={receiverEmail} onChange={setReceiverEmail} type="email" />
                    </div>}
                    <label className="mt-4 flex items-center justify-between gap-4 rounded-2xl border border-violet-200 bg-white px-4 py-3">
                      <span>
                        <span className="block text-sm font-black text-neutral-900">{documentType === 'NONE' ? 'Imprimir ticket al finalizar' : 'Imprimir al finalizar'}</span>
                        <span className="mt-1 block text-xs font-bold text-neutral-500">{documentType === 'NONE' ? 'Opcional. Actívalo solo si el cliente solicita el ticket.' : 'Abrirá el PDF oficial aceptado por SUNAT para elegir la impresora.'}</span>
                      </span>
                      <input type="checkbox" checked={printElectronicDocument} onChange={(event) => setPrintElectronicDocument(event.target.checked)} className="h-6 w-6 shrink-0 accent-violet-600" />
                    </label>
                  </div>
                )}
                <label className="block">
                  <span className="text-sm font-black text-neutral-700">Fecha de venta</span>
                  <input
                    type="date"
                    value={saleDate}
                    min={saleDateInputMinValue()}
                    max={saleDateInputMaxValue()}
                    onChange={(event) => setSaleDate(event.target.value)}
                    className="mt-2 w-full rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 font-bold text-neutral-950 outline-none transition placeholder:text-neutral-400 focus:border-amber-400"
                  />
                  {saleDate !== saleDateInputMaxValue() && (
                    <div className="mt-2 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs font-bold text-amber-800">
                      Esta venta se guardará con fecha anterior. Úsalo para regularizar ventas olvidadas.
                    </div>
                  )}
                </label>

                <InputField
                  label={isCourtesy ? 'Descuento aplicado por cortesia' : 'Descuento'}
                  value={isCourtesy ? effectiveDiscount.toFixed(2) : discount}
                  onChange={setDiscount}
                  type="number"
                  step="0.01"
                  prefix={getTenantCurrencySymbol()}
                  disabled={isCourtesy}
                />

                <div className="rounded-[24px] border border-emerald-200 bg-emerald-50 p-4">
                  <div className="text-xs font-black uppercase tracking-[0.18em] text-emerald-700">
                    Propina para {labels.professionalSingular}
                  </div>
                  <p className="mt-2 text-xs font-bold leading-5 text-emerald-800">
                    La propina se suma al total cobrado y aparecera en el calculo de pago al {labels.professionalSingular}.
                  </p>

                  <div className="mt-3 grid gap-3 sm:grid-cols-2">
                    <InputField
                      label="Monto de propina"
                      value={tipAmount}
                      onChange={setTipAmount}
                      type="number"
                      step="0.01"
                      prefix={getTenantCurrencySymbol()}
                    />

                    <SelectField
                      label={`${labels.professionalSingular[0].toUpperCase()}${labels.professionalSingular.slice(1)} que recibe la propina`}
                      value={tipBarberUserId || String(firstServiceBarberId || '')}
                      onChange={setTipBarberUserId}
                      options={[
                        { value: '', label: `Selecciona ${labels.professionalSingular}` },
                        ...barbers.map((barber) => ({
                          value: String(barber.id),
                          label: barber.name,
                        })),
                      ]}
                    />
                  </div>
                </div>

                {isCourtesy ? (
                  <div className="rounded-[24px] border border-amber-200 bg-amber-50 p-4">
                    <div className="text-xs font-black uppercase tracking-[0.18em] text-amber-700">
                      Venta gratis
                    </div>
                    <p className="mt-2 text-sm font-bold leading-6 text-amber-800">
                      No se registrara ningun cobro. La venta saldra como Gratis / Cortesia y aparecera en el resumen de {labels.courtesyPlural} por {labels.professionalSingular}.
                    </p>
                  </div>
                ) : (
                <div className="rounded-[28px] border border-neutral-200 bg-white p-5 shadow-[0_12px_30px_rgba(15,23,42,0.04)]">
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <div className="text-xs font-black uppercase tracking-[0.18em] text-amber-600">
                        Métodos de pago
                      </div>
                      <div className="mt-2 text-sm font-bold leading-5 text-neutral-500">
                        Divide el cobro entre los métodos configurados para este negocio.
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={addPaymentMethod}
                      className="rounded-2xl bg-neutral-950 px-5 py-3 text-sm font-black text-white transition hover:scale-[1.01] sm:shrink-0"
                    >
                      + Agregar método
                    </button>
                  </div>

                  <div className="mt-5 space-y-3">
                    {payments.map((payment, index) => (
                      <div
                        key={payment.key}
                        className="rounded-[22px] border border-neutral-200 bg-neutral-50 p-4"
                      >
                        <div className="mb-3 flex items-center justify-between gap-3">
                          <div className="text-xs font-black uppercase tracking-[0.16em] text-neutral-500">
                            {index === 0 ? 'Método principal' : `Método ${index + 1}`}
                          </div>

                          <button
                            type="button"
                            onClick={() => removePaymentMethod(index)}
                            disabled={payments.length <= 1}
                            className="rounded-xl border border-red-200 bg-white px-3 py-2 text-xs font-black text-red-700 transition hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            Quitar
                          </button>
                        </div>

                        <div className="grid gap-3 sm:grid-cols-[1fr_1fr]">
                          <SelectField
                            label="Método"
                            value={payment.method}
                            onChange={(value) => updatePayment(index, 'method', value)}
                            options={paymentOptions}
                          />

                          <InputField
                            label="Monto"
                            value={payment.amount}
                            onChange={(value) => updatePayment(index, 'amount', value)}
                            type="number"
                            step="0.01"
                            prefix={getTenantCurrencySymbol()}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className={`mt-5 rounded-[22px] border px-4 py-4 text-sm font-black ${
                    Math.abs(remainingPayment) <= 0.009
                      ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                      : remainingPayment > 0
                        ? 'border-amber-200 bg-amber-50 text-amber-700'
                        : 'border-red-200 bg-red-50 text-red-700'
                  }`}>
                    <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                      <span>Pagado: {formatMoney(paymentsTotal)}</span>
                      <span>{Math.abs(remainingPayment) <= 0.009 ? 'Pago completo' : remainingPayment > 0 ? `Falta ${formatMoney(remainingPayment)}` : `Sobra ${formatMoney(Math.abs(remainingPayment))}`}</span>
                    </div>
                  </div>
                </div>
                )}

                {!isCourtesy && cashPaymentTotal > 0 && (
                  <InputField
                    label={`Efectivo recibido para ${formatMoney(cashPaymentTotal)}`}
                    value={cashReceived}
                    onChange={setCashReceived}
                    type="number"
                    step="0.01"
                    prefix={getTenantCurrencySymbol()}
                  />
                )}
              </div>

              <div className="mt-5 grid gap-3">
                <StatCard title="Subtotal" value={formatMoney(subtotal)} />
                <StatCard title="Descuento" value={formatMoney(effectiveDiscount)} tone={isCourtesy ? 'gold' : 'default'} />
                <StatCard title="Propina" value={formatMoney(tipNumber)} tone={tipNumber > 0 ? 'green' : 'default'} />
                <StatCard title="Pagado" value={formatMoney(isCourtesy ? 0 : paymentsTotal)} tone={isCourtesy || Math.abs(remainingPayment) <= 0.009 ? 'green' : 'gold'} />
                <StatCard title="Vuelto" value={formatMoney(change)} tone={change > 0 ? 'green' : 'default'} />
              </div>
            </div>
          </div>


          <div className="sticky bottom-0 z-30 -mx-5 -mb-4 mt-1 border-t border-neutral-200 bg-white/95 px-5 py-4 shadow-[0_-18px_45px_rgba(15,23,42,0.10)] backdrop-blur-xl sm:-mx-6 sm:-mb-5 sm:px-6">
            {errorMsg && <div className="mb-3"><ErrorBox message={errorMsg} /></div>}
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
              <div className="flex items-center justify-between rounded-2xl bg-neutral-950 px-4 py-3 text-white sm:min-w-[220px]">
                <span className="text-xs font-black uppercase tracking-[0.16em] text-white/50">Total</span>
                <span className="text-xl font-black">{formatMoney(total)}</span>
              </div>
              <button
                disabled={saving || loadingAllowedServices}
                className="min-h-14 flex-1 rounded-2xl bg-amber-400 px-5 py-4 font-black text-neutral-950 shadow-[0_12px_28px_rgba(251,191,36,0.28)] transition hover:bg-amber-300 disabled:opacity-60"
              >
                {saving
                  ? documentType === 'INVOICE'
                    ? 'Emitiendo factura...'
                    : documentType === 'RECEIPT'
                      ? 'Emitiendo boleta...'
                      : 'Guardando venta...'
                  : isCourtesy
                    ? 'Guardar cortesía gratis'
                    : documentType === 'INVOICE'
                      ? 'Cobrar y emitir factura' + (printElectronicDocument ? ' · Imprimir' : '')
                      : documentType === 'RECEIPT'
                        ? 'Cobrar y emitir boleta' + (printElectronicDocument ? ' · Imprimir' : '')
                        : 'Cobrar y guardar venta' + (printElectronicDocument ? ' · Imprimir ticket' : '')}
              </button>
            </div>
          </div>
        </form>
      )}
    </ModalShell>
  );
}

function ModalShell({ title, subtitle, onClose, children }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-neutral-950/60 p-2 backdrop-blur-sm sm:p-4">
      <div className="flex max-h-[96vh] w-full max-w-7xl flex-col overflow-hidden rounded-[30px] border border-white/10 bg-white shadow-[0_30px_90px_rgba(15,23,42,0.35)] sm:max-h-[94vh] sm:rounded-[34px]">
        <div className="flex shrink-0 items-start justify-between gap-4 border-b border-neutral-200 bg-white px-5 py-4 sm:px-6 sm:py-5">
          <div>
            <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-600">{subtitle}</div>
            <h2 className="mt-1 text-2xl font-black text-neutral-950">{title}</h2>
          </div>

          <button
            onClick={onClose}
            className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-2 text-sm font-black text-neutral-700 hover:bg-neutral-100"
          >
            Cerrar
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4 sm:px-6 sm:py-5">
          {children}
        </div>
      </div>
    </div>
  );
}

function InputField({ label, value, onChange, type = 'text', step, prefix, disabled = false }) {
  return (
    <label className="block">
      <span className="text-sm font-black text-neutral-700">{label}</span>

      <div className="mt-2 flex rounded-2xl border border-neutral-200 bg-neutral-50 focus-within:border-amber-400">
        {prefix && (
          <div className="flex items-center px-4 font-black text-neutral-500">{prefix}</div>
        )}

        <input
          type={type}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          className="w-full rounded-2xl bg-transparent px-4 py-4 font-bold text-neutral-950 outline-none disabled:cursor-not-allowed disabled:text-neutral-400"
        />
      </div>
    </label>
  );
}

function TextAreaField({ label, value, onChange, placeholder }) {
  return (
    <label className="block">
      <span className="text-sm font-black text-neutral-700">{label}</span>

      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={4}
        className="mt-2 w-full rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 font-bold text-neutral-950 outline-none transition placeholder:text-neutral-400 focus:border-amber-400"
      />
    </label>
  );
}

function SelectField({ label, value, onChange, options, disabled = false }) {
  return (
    <label className="block">
      <span className="text-sm font-black text-neutral-700">{label}</span>

      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="mt-2 w-full rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4 font-bold text-neutral-950 outline-none transition focus:border-amber-400"
      >
        {options.map((item) => (
          <option key={item.value} value={item.value}>
            {item.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function ErrorBox({ message }) {
  return <PremiumErrorState message={message} />;
}
function cashStatusLabel(status) {
  const code = String(status || '').toUpperCase();

  if (code === 'OPEN') return 'Abierta';
  if (code === 'CLOSED') return 'Cerrada';
  if (code === 'AUTO_CLOSED') return 'Cierre automático';

  return status || 'Sin estado';
}

function cashStatusClass(status) {
  const code = String(status || '').toUpperCase();

  if (code === 'OPEN') return 'border-emerald-200 bg-emerald-50 text-emerald-700';
  if (code === 'CLOSED') return 'border-blue-200 bg-blue-50 text-blue-700';
  if (code === 'AUTO_CLOSED') return 'border-orange-200 bg-orange-50 text-orange-700';

  return 'border-neutral-200 bg-neutral-50 text-neutral-700';
}

function paymentTotalFrom(cash, key) {
  const source =
    Array.isArray(cash?.paymentMethodBalances) && cash.paymentMethodBalances.length > 0
      ? cash.paymentMethodBalances
      : cash?.paymentMethodsSummary || [];

  const normalizedKey = normalizeMethod(key);

  return source.reduce((sum, item) => {
    const code = normalizeMethod(item?.paymentMethod);
    return code === normalizedKey ? sum + Number(item?.totalAmount || 0) : sum;
  }, 0);
}

function HistorySummaryCard({ items, paymentMethods = DEFAULT_PAYMENT_METHODS }) {
  const totals = items.reduce(
    (acc, cash) => {
      acc.sales += Number(cash.salesTotal || 0);
      acc.income += Number(cash.movementsIncome || 0);
      acc.expense += Number(cash.movementsExpense || 0);
      acc.cashExpense += Number(cash.cashMovementsExpense || 0);
      acc.expected += Number(cash.closingAmountExpected || 0);

      const source = Array.isArray(cash?.paymentMethodBalances) && cash.paymentMethodBalances.length > 0
        ? cash.paymentMethodBalances
        : cash?.paymentMethodsSummary || [];

      source.forEach((item) => {
        const code = summaryMethodOf(item);
        if (!code || code === 'FREE' || code === 'DEPOSIT_APPLIED') return;
        const current = acc.methods.get(code) || { code, totalAmount: 0, count: 0 };
        current.totalAmount += summaryAmountOf(item);
        current.count += summaryCountOf(item);
        acc.methods.set(code, current);
      });

      return acc;
    },
    {
      sales: 0,
      income: 0,
      expense: 0,
      cashExpense: 0,
      expected: 0,
      methods: new Map(),
    }
  );

  const net = totals.sales + totals.income - totals.expense;
  const otherExpense = Math.max(0, totals.expense - totals.cashExpense);
  const methodRows = buildPaymentMethodRows(
    Array.from(totals.methods.values()).map((item) => ({
      paymentMethod: item.code,
      totalAmount: item.totalAmount,
      count: item.count,
    })),
    Array.from(totals.methods.values()).map((item) => ({
      paymentMethod: item.code,
      totalAmount: item.totalAmount,
      count: item.count,
    })),
    paymentMethods
  );

  return (
    <div className="rounded-[30px] border border-amber-200 bg-[linear-gradient(135deg,#FFFBEB_0%,#FFFFFF_68%)] p-5 shadow-[0_16px_38px_rgba(15,23,42,0.045)]">
      <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-700">
            Resumen del rango
          </div>
          <div className="mt-1 text-sm font-semibold text-neutral-500">
            Consolidado de cajas encontradas en el rango seleccionado.
          </div>
        </div>

        <div className="rounded-full bg-white px-4 py-2 text-xs font-black text-neutral-700 shadow-sm">
          {items.length} caja{items.length === 1 ? '' : 's'}
        </div>
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-7">
        <HistoryMetricCard label="Ventas" value={totals.sales} tone="green" />
        <HistoryMetricCard label="Ingresos" value={totals.income} />
        <HistoryMetricCard label="Salidas totales" value={totals.expense} tone="red" />
        <HistoryMetricCard label="Salidas en efectivo" value={totals.cashExpense} tone="red" />
        <HistoryMetricCard label="Salidas otros métodos" value={otherExpense} tone="red" />
        <HistoryMetricCard label="Neto registrado" value={net} tone={net >= 0 ? 'green' : 'red'} />
        <HistoryMetricCard label="Efectivo esperado" value={totals.expected} tone={totals.expected >= 0 ? 'green' : 'red'} />
      </div>

      <div className="mt-5 rounded-[24px] border border-neutral-200 bg-white p-4">
        <div className="mb-3 text-xs font-black uppercase tracking-[0.18em] text-neutral-500">
          Saldo por método
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {methodRows.map((method) => (
            <HistoryPaymentPill
              key={method.code}
              label={method.label || methodLabel(method.code)}
              value={method.balanceAmount}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function HistoryMetricCard({ label, value, tone = 'default' }) {
  const isRed = tone === 'red';
  const isGreen = tone === 'green';

  const classes = isRed
    ? 'border-red-200 bg-red-50 text-red-700'
    : isGreen
      ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
      : 'border-neutral-200 bg-white text-neutral-950';

  return (
    <div className={`min-w-0 rounded-[22px] border p-4 ${classes}`}>
      <div className="text-sm font-black text-neutral-500">{label}</div>
      <div className="mt-2 truncate text-2xl font-black">
        {formatMoney(value)}
      </div>
    </div>
  );
}

function HistoryPaymentPill({ label, value }) {
  return (
    <div className="min-w-0 rounded-2xl border border-neutral-200 bg-white px-4 py-3">
      <div className="truncate text-xs font-bold text-neutral-500">{label}</div>
      <div className={`mt-1 truncate text-base font-black ${amountClassByValue(value)}`}>
        {formatMoney(value)}
      </div>
    </div>
  );
}

function FundHistorySection({ items = [], summary = null, onEdit, onDelete }) {
  const totalIn = items.filter((item) => Number(item.signedAmount || 0) >= 0).reduce((sum, item) => sum + Number(item.amount || 0), 0);
  const totalOut = items.filter((item) => Number(item.signedAmount || 0) < 0).reduce((sum, item) => sum + Number(item.amount || 0), 0);
  const net = items.reduce((sum, item) => sum + Number(item.signedAmount || 0), 0);
  const openingBalance = Number(summary?.openingBalance || 0);
  const closingBalance = Number(summary?.closingBalance ?? (openingBalance + net));
  const typeLabel = (type) => ({
    CLOSING_DEPOSIT: 'Ingreso por cierre',
    OPENING_WITHDRAWAL: 'Retiro para apertura',
    MANUAL_DEPOSIT: 'Ingreso manual',
    MANUAL_WITHDRAWAL: 'Retiro manual',
    EXPENSE: 'Gasto desde fondo',
    ADJUSTMENT_IN: 'Ajuste positivo',
    ADJUSTMENT_OUT: 'Ajuste negativo',
  }[String(type || '').toUpperCase()] || 'Movimiento de fondo');

  return (
    <section className="overflow-hidden rounded-[28px] border border-amber-400/20 bg-[linear-gradient(135deg,#111827_0%,#201A2C_100%)] p-5 text-white shadow-[0_18px_44px_rgba(15,23,42,0.14)]">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="text-[10px] font-black uppercase tracking-[0.22em] text-amber-300">Fondo acumulado</div>
          <h3 className="mt-1 text-xl font-black">Movimientos de gestión de fondos</h3>
          <p className="mt-1 text-sm font-semibold text-slate-300">Ingresos y retiros separados de la caja del día.</p>
        </div>
        <span className="w-fit rounded-full bg-white/10 px-3 py-1.5 text-xs font-black">{items.length} movimientos</span>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <div className="rounded-2xl border border-white/10 bg-white/[0.06] p-4"><p className="text-[10px] font-black uppercase tracking-wider text-slate-400">Saldo inicial</p><p className="mt-2 text-xl font-black text-slate-200">{formatMoney(openingBalance)}</p></div>
        <div className="rounded-2xl border border-white/10 bg-white/[0.06] p-4"><p className="text-[10px] font-black uppercase tracking-wider text-slate-400">Ingresos del rango</p><p className="mt-2 text-xl font-black text-emerald-300">{formatMoney(totalIn)}</p></div>
        <div className="rounded-2xl border border-white/10 bg-white/[0.06] p-4"><p className="text-[10px] font-black uppercase tracking-wider text-slate-400">Retiros del rango</p><p className="mt-2 text-xl font-black text-red-300">{formatMoney(totalOut)}</p></div>
        <div className="rounded-2xl border border-white/10 bg-white/[0.06] p-4"><p className="text-[10px] font-black uppercase tracking-wider text-slate-400">Movimientos del rango</p><p className={`mt-2 text-xl font-black ${net >= 0 ? 'text-amber-300' : 'text-red-300'}`}>{net >= 0 ? '+' : ''}{formatMoney(net)}</p></div>
        <div className="rounded-2xl border border-emerald-400/20 bg-emerald-400/[0.08] p-4"><p className="text-[10px] font-black uppercase tracking-wider text-emerald-200">Saldo final</p><p className="mt-2 text-xl font-black text-emerald-300">{formatMoney(closingBalance)}</p></div>
      </div>
      <p className="mt-3 text-xs font-bold text-slate-400">Saldo inicial + movimientos del rango = saldo final. El filtro solo afecta el periodo seleccionado.</p>

      <div className="mt-5 max-h-[360px] overflow-auto rounded-2xl border border-white/10">
        <table className="w-full min-w-[860px] text-left text-sm">
          <thead className="sticky top-0 bg-slate-950 text-[10px] font-black uppercase tracking-wider text-slate-400">
            <tr><th className="px-4 py-3">Fecha</th><th className="px-4 py-3">Movimiento</th><th className="px-4 py-3">Método</th><th className="px-4 py-3">Responsable / caja</th><th className="px-4 py-3 text-right">Monto</th></tr>
          </thead>
          <tbody className="divide-y divide-white/10">
            {items.map((item) => {
              const income = Number(item.signedAmount || 0) >= 0;
              const editable = ['MANUAL_DEPOSIT', 'MANUAL_WITHDRAWAL', 'ADJUSTMENT_IN', 'ADJUSTMENT_OUT'].includes(String(item.type || '').toUpperCase()) && !item.cashRegisterId;
              return (
                <tr key={item.id} className="align-top hover:bg-white/[0.04]">
                  <td className="px-4 py-3 font-bold text-slate-300">{formatDateTime(item.movementDate || item.date)}</td>
                  <td className="px-4 py-3"><p className="font-black">{item.concept || typeLabel(item.type)}</p><p className="mt-1 text-xs font-semibold text-slate-400">{typeLabel(item.type)}{item.note ? ` · ${item.note}` : ''}</p></td>
                  <td className="px-4 py-3 font-bold text-blue-300">{methodLabel(item.paymentMethod)}</td>
                  <td className="px-4 py-3"><p className="font-bold">{item.actorUserName || 'Sistema'}</p><p className="mt-1 text-xs text-slate-400">{item.cashRegisterId ? `Caja #${item.cashRegisterId}` : 'Sin caja relacionada'}</p></td>
                  <td className={`px-4 py-3 text-right font-black ${income ? 'text-emerald-300' : 'text-red-300'}`}><div>{income ? '+' : '-'}{formatMoney(Math.abs(Number(item.amount || 0)))}</div>{editable ? <div className="mt-2 flex justify-end gap-2"><button type="button" onClick={() => onEdit?.(item)} className="rounded-lg bg-amber-300/15 px-2 py-1 text-[10px] text-amber-200">Editar</button><button type="button" onClick={() => onDelete?.(item)} className="rounded-lg bg-red-400/15 px-2 py-1 text-[10px] text-red-200">Eliminar</button></div> : <div className="mt-1 text-[9px] text-slate-500">Automático</div>}</td>
                </tr>
              );
            })}
            {items.length === 0 && <tr><td colSpan="5" className="px-4 py-8 text-center font-bold text-slate-400">No hubo movimientos del fondo en este rango.</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function HistoryDetailModal({ branch, cash, from, to, paymentMethods: initialPaymentMethods = DEFAULT_PAYMENT_METHODS, session = null, canManageFund = false, labels = readBusinessLabels(), onClose }) {
  const [sales, setSales] = useState([]);
  const [movements, setMovements] = useState(Array.isArray(cash?.movements) ? cash.movements : []);
  const [editingSale, setEditingSale] = useState(null);
  const [editingMovement, setEditingMovement] = useState(null);
  const [deletingSaleId, setDeletingSaleId] = useState(null);
  const [deletingMovementId, setDeletingMovementId] = useState(null);
  const [paymentMethods] = useState(initialPaymentMethods);
  const [loadingSales, setLoadingSales] = useState(true);
  const [loadingMovements, setLoadingMovements] = useState(false);
  const [auditLogs, setAuditLogs] = useState([]);
  const [loadingAudit, setLoadingAudit] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const visibleHistoryMovements = movements.filter(
    (movement) =>
      canManageFund ||
      String(movement?.fundingSource || '').toUpperCase() !==
        'ACCUMULATED_FUND'
  );

  async function loadSales() {
    setLoadingSales(true);
    setErrorMsg('');

    try {
      const data = await getSalesByCashRegister({
        branchId: branch.id,
        cashRegisterId: cash.id,
      });

      setSales(data);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudieron cargar las ventas de esta caja.');
    } finally {
      setLoadingSales(false);
    }
  }

  async function loadMovements() {
    setLoadingMovements(true);
    setErrorMsg('');

    try {
      const data = await getCashMovements({
        branchId: branch.id,
        cashRegisterId: cash.id,
        from,
        to,
      });

      setMovements(Array.isArray(data) ? data : []);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudieron cargar los movimientos de esta caja.');
    } finally {
      setLoadingMovements(false);
    }
  }

  async function loadAudit() {
    setLoadingAudit(true);

    try {
      const openedDate = cash?.openedAt ? new Date(cash.openedAt) : new Date();
      const from = Number.isNaN(openedDate.getTime()) ? toDateInputValue(new Date()) : toDateInputValue(openedDate);
      const to = toDateInputValue(new Date());
      const data = await getCashAudit({
        branchId: branch.id,
        cashRegisterId: cash.id,
        from,
        to,
      });

      setAuditLogs(Array.isArray(data) ? data : []);
    } catch (_) {
      setAuditLogs([]);
    } finally {
      setLoadingAudit(false);
    }
  }
  useEffect(() => {
    setMovements(Array.isArray(cash?.movements) ? cash.movements : []);
    loadSales();
    loadMovements();
    loadAudit();
  }, [branch.id, cash.id]);
  const paymentsSource =
    cash?.paymentMethodBalances?.length > 0
      ? cash.paymentMethodBalances
      : cash?.paymentMethodsSummary || [];

  async function handleDeleteHistorySale(sale) {
    const saleId = saleIdOf(sale);
    if (!saleId) return;
    const auditReason = window.prompt('Motivo para eliminar la venta #' + saleId + ':');
    if (auditReason === null) return;
    if (!auditReason.trim()) {
      setErrorMsg('El motivo de auditoria es obligatorio para eliminar una venta.');
      return;
    }
    if (!await premiumConfirm('Eliminar la venta #' + saleId + '? Esta accion no se puede deshacer.')) return;

    setDeletingSaleId(saleId);
    setErrorMsg('');

    try {
      await deleteCashSale({ branchId: branch.id, saleId, auditReason: auditReason.trim() });
      await loadSales();
      await loadAudit();
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo eliminar la venta.');
    } finally {
      setDeletingSaleId(null);
    }
  }

  async function handleDeleteHistoryMovement(movement) {
    const movementId = movement?.id;
    if (!movementId) return;
    const auditReason = window.prompt('Motivo para eliminar el movimiento #' + movementId + ':');
    if (auditReason === null) return;
    if (!auditReason.trim()) {
      setErrorMsg('El motivo de auditoria es obligatorio para eliminar un movimiento.');
      return;
    }
    if (!await premiumConfirm('Eliminar el movimiento #' + movementId + '? Esta accion no se puede deshacer.')) return;

    setDeletingMovementId(movementId);
    setErrorMsg('');

    try {
      await deleteCashMovement({ branchId: branch.id, movementId, auditReason: auditReason.trim() });
      await loadMovements();
      await loadAudit();
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo eliminar el movimiento.');
    } finally {
      setDeletingMovementId(null);
    }
  }

  return (
    <ModalShell title="Detalle de caja" subtitle={branch?.name || 'Sede'} onClose={onClose}>
      <div className="space-y-5">
        <div className="flex flex-wrap items-center gap-3">
          <div className={`rounded-full border px-4 py-2 text-xs font-black ${cashStatusClass(cash.status)}`}>
            {cashStatusLabel(cash.status)}
          </div>

          <div className="rounded-full border border-neutral-200 bg-neutral-50 px-4 py-2 text-xs font-black text-neutral-600">
            Caja #{cash.id}
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <StatCard title="Apertura" value={formatDateTime(cash.openedAt)} helper={cash.openedByUserName || 'Usuario no registrado'} />
          <StatCard title="Cierre" value={formatDateTime(cash.closedAt)} helper={cash.closedByUserName || cash.assignedUserName || 'Pendiente'} />
          <StatCard title="Ventas total" value={formatMoney(cash.salesTotal)} helper="Total registrado" tone="gold" />
          <StatCard title="Esperado" value={formatMoney(cash.closingAmountExpected)} helper="Efectivo fisico esperado" tone={balanceTone(cash.closingAmountExpected)} />
          <StatCard title="Contado" value={formatMoney(cash.closingAmountCounted)} helper="Monto contado al cierre" />
          <StatCard title="Diferencia" value={formatMoney(cash.differenceAmount)} helper="Contado - esperado" tone={Number(cash.differenceAmount || 0) === 0 ? 'green' : 'red'} />
        </div>

        <div className="rounded-[26px] border border-neutral-200 bg-white p-5">
          <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-600">
            Metodos de pago
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {paymentsSource.length === 0 ? (
              <div className="rounded-2xl bg-neutral-50 p-4 text-sm font-bold text-neutral-500">
                Sin metodos registrados.
              </div>
            ) : (
              paymentsSource.map((payment) => (
                <div
                  key={`${payment.paymentMethod}-${payment.totalAmount}`}
                  className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-4"
                >
                  <div className="text-sm font-black text-neutral-950">
                    {paymentLabelFromOptions(paymentMethods, payment.paymentMethod)}
                  </div>
                  <div className="mt-1 text-xs text-neutral-500">
                    {payment.count || 0} operacion(es)
                  </div>
                  <div className={`mt-2 text-lg font-black ${amountClassByValue(payment.totalAmount)}`}>
                    {formatMoney(payment.totalAmount)}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="rounded-[26px] border border-neutral-200 bg-white p-5">
          <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-600">
            Ventas de esta caja
          </div>

          <div className="mt-4 overflow-x-auto rounded-[22px] border border-neutral-200">
            <table className="w-full text-left text-sm">
              <thead className="bg-neutral-950 text-white">
                <tr>
                  <th className="px-4 py-3 font-black">Cliente</th>
                  <th className="px-4 py-3 font-black">{labels.professionalDisplay}</th>
                  <th className="px-4 py-3 font-black">Metodo</th>
                  <th className="px-4 py-3 font-black">Total</th>
                  <th className="px-4 py-3 font-black">Fecha</th>
                  <th className="px-4 py-3 font-black text-right">Acciones</th>
                </tr>
              </thead>
              <tbody>
                {loadingSales ? (
                  <tr>
                    <td className="px-4 py-5 text-neutral-500" colSpan="6">
                      Cargando ventas...
                    </td>
                  </tr>
                ) : errorMsg ? (
                  <tr>
                    <td className="px-4 py-5 text-red-600" colSpan="6">
                      {errorMsg}
                    </td>
                  </tr>
                ) : sales.length === 0 ? (
                  <tr>
                    <td className="px-4 py-5 text-neutral-500" colSpan="6">
                      No hubo ventas en esta caja.
                    </td>
                  </tr>
                ) : (
                  sales.map((sale) => {
                    const saleId = saleIdOf(sale);
                    const isDeleting = deletingSaleId === saleId;

                    return (
                      <tr key={saleId} className="border-t border-neutral-200">
                        <td className="px-4 py-4 font-black text-neutral-950">
                          {sale.customerName || 'Cliente ocasional'}
                        </td>
                        <td className="px-4 py-4 font-bold text-neutral-700">
                          {saleBarberName(sale)}
                        </td>
                        <td className="px-4 py-4 font-bold text-neutral-700">
                          {methodLabel(sale.metodoPago)}
                        </td>
                        <td className="px-4 py-4 font-black text-emerald-700">
                          {formatMoney(sale.total)}
                        </td>
                        <td className="px-4 py-4 text-xs font-bold text-neutral-500">
                          {formatDateTime(saleDateOf(sale))}
                        </td>
                        <td className="px-4 py-4">
                          <div className="flex justify-end gap-2">
                            <button type="button" onClick={() => setEditingSale(sale)} className="rounded-xl border border-neutral-200 bg-white px-3 py-2 text-xs font-black text-neutral-700 hover:bg-neutral-100">
                              Editar
                            </button>
                            <button type="button" onClick={() => handleDeleteHistorySale(sale)} disabled={isDeleting} className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs font-black text-red-700 hover:bg-red-100 disabled:opacity-60">
                              {isDeleting ? 'Eliminando...' : 'Eliminar'}
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="rounded-[26px] border border-neutral-200 bg-white p-5">
          <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-600">
            Movimientos
          </div>

          <div className="mt-4 space-y-3">
            {loadingMovements ? (
              <div className="rounded-2xl bg-neutral-50 p-4 text-sm font-bold text-neutral-500">
                Cargando movimientos...
              </div>
            ) : visibleHistoryMovements.length === 0 ? (
              <div className="rounded-2xl bg-neutral-50 p-4 text-sm font-bold text-neutral-500">
                No hubo gastos, ingresos, adelantos ni pagos.
              </div>
            ) : (
              visibleHistoryMovements.map((movement) => {
                const isDeleting = deletingMovementId === movement.id;

                return (
                  <div key={movement.id} className="rounded-2xl border border-neutral-200 bg-neutral-50 p-4">
                    <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                      <div>
                        <div className="font-black text-neutral-950">
                          {movement.concept || movementTypeLabel(movement.type)}
                        </div>
                        <div className="mt-1 text-xs font-bold text-neutral-500">
                          {movementTypeLabel(movement.type)} - {methodLabel(movement.paymentMethod)} - {formatDateTime(movement.movementDate)}
                        </div>
                        {movement.barberUserName && (
                          <div className="mt-1 text-xs font-bold text-amber-700">
                            {labels.professionalDisplay}: {movement.barberUserName}
                          </div>
                        )}
                        {movement.note && (
                          <div className="mt-1 text-xs text-neutral-500">
                            {movement.note}
                          </div>
                        )}
                      </div>

                      <div className="flex flex-col items-start gap-2 lg:items-end">
                        <div className={`font-black ${movementAmountClass(movement.type)}`}>
                          {formatMoney(movement.amount)}
                        </div>
                        <div className="flex gap-2">
                          <button type="button" onClick={() => setEditingMovement(movement)} className="rounded-xl border border-neutral-200 bg-white px-3 py-2 text-xs font-black text-neutral-700 hover:bg-neutral-100">
                            Editar
                          </button>
                          <button type="button" onClick={() => handleDeleteHistoryMovement(movement)} disabled={isDeleting} className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs font-black text-red-700 hover:bg-red-100 disabled:opacity-60">
                            {isDeleting ? 'Eliminando...' : 'Eliminar'}
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

        <div className="rounded-[26px] border border-neutral-200 bg-white p-5">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-600">
              Actividad de auditoria
            </div>
            <div className="text-xs font-bold text-neutral-400">
              Ediciones y eliminaciones
            </div>
          </div>

          <div className="mt-4 space-y-3">
            {loadingAudit ? (
              <div className="rounded-2xl bg-neutral-50 p-4 text-sm font-bold text-neutral-500">
                Cargando actividad...
              </div>
            ) : auditLogs.length === 0 ? (
              <div className="rounded-2xl bg-neutral-50 p-4 text-sm font-bold text-neutral-500">
                Aun no hay cambios auditados en esta caja.
              </div>
            ) : (
              auditLogs.map((log) => (
                <div key={log.id} className="rounded-2xl border border-amber-100 bg-amber-50/50 p-4">
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <div className="text-sm font-black text-neutral-950">
                        {auditActionLabel(log.action)} {auditEntityLabel(log.entityType)} #{log.entityId || '-'}
                      </div>
                      <div className="mt-1 text-xs font-bold text-neutral-500">
                        {formatDateTime(log.createdAt)} - {log.actorUserName || 'Sistema'}
                      </div>
                      {log.reason && (
                        <div className="mt-2 rounded-xl bg-white px-3 py-2 text-xs font-bold text-neutral-700">
                          Motivo: {log.reason}
                        </div>
                      )}
                    </div>
                    <details className="text-xs font-bold text-neutral-500">
                      <summary className="cursor-pointer select-none rounded-xl border border-neutral-200 bg-white px-3 py-2 text-neutral-700">
                        Ver datos
                      </summary>
                      <div className="mt-2 grid gap-2 lg:grid-cols-2">
                        <pre className="max-h-48 overflow-auto rounded-xl bg-neutral-950 p-3 text-[11px] text-white">
                          {log.beforeSnapshot || 'Sin datos previos'}
                        </pre>
                        <pre className="max-h-48 overflow-auto rounded-xl bg-neutral-950 p-3 text-[11px] text-white">
                          {log.afterSnapshot || 'Sin datos nuevos'}
                        </pre>
                      </div>
                    </details>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      {editingSale && (
        <EditSaleModal
          branch={branch}
          sale={editingSale}
          paymentMethods={paymentMethods}
          session={session}
          onClose={() => setEditingSale(null)}
          onSaved={async () => {
            setEditingSale(null);
            await loadSales();
            await loadAudit();
          }}
        />
      )}

      {editingMovement && (
        <MovementModal
          branch={branch}
          cashRegister={cash}
          paymentMethods={paymentMethods}
          canManageFund={canManageFund}
          initialMovement={editingMovement}
          onClose={() => setEditingMovement(null)}
          onSaved={async () => {
            setEditingMovement(null);
            await loadMovements();
            await loadAudit();
          }}
        />
      )}
    </ModalShell>
  );
}

function CashHistoryModal({ branch, paymentMethods = DEFAULT_PAYMENT_METHODS, session = null, canManageFund = false, onClose }) {
  const today = new Date();
  const fromDefault = new Date();
  fromDefault.setDate(today.getDate() - 30);

  const [from, setFrom] = useState(toDateInputValue(fromDefault));
  const [to, setTo] = useState(toDateInputValue(today));
  const [items, setItems] = useState([]);
  const [fundMovements, setFundMovements] = useState([]);
  const [fundSummary, setFundSummary] = useState(null);
  const [selectedCash, setSelectedCash] = useState(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState('');
  const [errorMsg, setErrorMsg] = useState('');

  async function loadHistory() {
    setLoading(true);
    setErrorMsg('');

    try {
      const [data, fundData, fundSummaryData] = await Promise.all([
        getCashHistory({ branchId: branch.id, from, to }),
        canManageFund
          ? getCashFundMovements(branch.id, { from, to })
          : Promise.resolve([]),
        canManageFund
          ? getCashFundMovementSummary(branch.id, { from, to })
          : Promise.resolve(null),
      ]);

      setItems(Array.isArray(data) ? data : []);
      setFundMovements(Array.isArray(fundData) ? fundData : []);
      setFundSummary(fundSummaryData);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo cargar el historial.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadHistory();
  }, []);

  async function exportHistory(format) {
    setExporting(format);
    setErrorMsg('');
    try {
      const details = await Promise.all(items.map(async (cash) => {
        const [sales, movements] = await Promise.all([
          getSalesByCashRegister({ branchId: branch.id, cashRegisterId: cash.id }),
          getCashMovements({ branchId: branch.id, cashRegisterId: cash.id, from, to }),
        ]);
        return { cash, sales, movements };
      }));
      const payload = { branch, items, details, fundMovements, fundSummary, from, to, formatDateTime, statusLabel: cashStatusLabel };
      if (format === 'excel') exportCashHistoryExcel(payload);
      else await exportCashHistoryPdf(payload);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo exportar el historial.');
    } finally {
      setExporting('');
    }
  }

  async function editFundMovement(item) {
    const amount = window.prompt('Monto corregido:', String(item.amount ?? ''));
    if (amount === null) return;
    const parsed = Number(String(amount).replace(',', '.'));
    if (!Number.isFinite(parsed) || parsed <= 0) { setErrorMsg('Ingresa un monto válido mayor a cero.'); return; }
    const concept = window.prompt('Concepto:', item.concept || '');
    if (concept === null || !concept.trim()) return;
    const note = window.prompt('Detalle adicional:', item.note || '');
    if (note === null) return;
    const auditReason = window.prompt('Motivo obligatorio de auditoría:');
    if (auditReason === null || !auditReason.trim()) { setErrorMsg('El motivo de auditoría es obligatorio.'); return; }
    try {
      await updateCashFundMovement({ branchId: branch.id, movementId: item.id, type: item.type, paymentMethod: item.paymentMethod, amount: parsed, concept: concept.trim(), note: note.trim() || null, movementDate: item.movementDate?.slice?.(0, 10) || null, auditReason: auditReason.trim() });
      await loadHistory();
    } catch (error) { setErrorMsg(error.message || 'No se pudo actualizar el movimiento del fondo.'); }
  }

  async function removeFundMovement(item) {
    const auditReason = window.prompt('Motivo obligatorio para eliminar este movimiento:');
    if (auditReason === null || !auditReason.trim()) { setErrorMsg('El motivo de auditoría es obligatorio.'); return; }
    if (!await premiumConfirm(`¿Eliminar “${item.concept || 'Movimiento'}”? El cambio quedará auditado.`)) return;
    try {
      await deleteCashFundMovement({ branchId: branch.id, movementId: item.id, auditReason: auditReason.trim() });
      await loadHistory();
    } catch (error) { setErrorMsg(error.message || 'No se pudo eliminar el movimiento del fondo.'); }
  }
  return (
    <>
      <ModalShell title="Historial de caja" subtitle={branch?.name || 'Sede'} onClose={onClose}>
        <div className="space-y-5">
          <div className="grid gap-3 lg:grid-cols-[1fr_1fr_130px_130px_130px]">
            <InputField label="Desde" value={from} onChange={setFrom} type="date" />
            <InputField label="Hasta" value={to} onChange={setTo} type="date" />
            <button
              type="button"
              onClick={loadHistory}
              className="self-end rounded-2xl bg-neutral-950 px-5 py-4 text-sm font-black text-white transition hover:scale-[1.01]"
            >
              Buscar
            </button>
            <button type="button" onClick={() => exportHistory('excel')} disabled={loading || (items.length === 0 && fundMovements.length === 0) || Boolean(exporting)} className="self-end rounded-2xl border border-emerald-200 bg-emerald-50 px-5 py-4 text-sm font-black text-emerald-800 disabled:opacity-50">{exporting === 'excel' ? 'Generando...' : 'Excel'}</button>
            <button type="button" onClick={() => exportHistory('pdf')} disabled={loading || (items.length === 0 && fundMovements.length === 0) || Boolean(exporting)} className="self-end rounded-2xl border border-red-200 bg-red-50 px-5 py-4 text-sm font-black text-red-700 disabled:opacity-50">{exporting === 'pdf' ? 'Generando...' : 'PDF'}</button>
          </div>

          {!loading && !errorMsg && items.length > 0 && (
            <HistorySummaryCard items={items} paymentMethods={paymentMethods} />
          )}

          {!loading && !errorMsg && canManageFund && (
            <FundHistorySection items={fundMovements} summary={fundSummary} onEdit={editFundMovement} onDelete={removeFundMovement} />
          )}

          {loading ? (
            <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-5 text-sm font-black text-neutral-500">
              Cargando historial...
            </div>
          ) : errorMsg ? (
            <ErrorBox message={errorMsg} />
          ) : items.length === 0 ? (
            <EmptyCard
              title="Sin cajas en este rango"
              text="No se encontraron cajas registradas para las fechas seleccionadas."
            />
          ) : (
            <div className="space-y-3">
              {items.map((cash) => (
                <button
                  key={cash.id}
                  type="button"
                  onClick={() => setSelectedCash(cash)}
                  className="w-full rounded-[26px] border border-neutral-200 bg-white p-5 text-left shadow-[0_10px_28px_rgba(15,23,42,0.045)] transition hover:-translate-y-0.5 hover:shadow-[0_16px_38px_rgba(15,23,42,0.08)]"
                >
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <span className={`rounded-full border px-3 py-1 text-xs font-black ${cashStatusClass(cash.status)}`}>
                          {cashStatusLabel(cash.status)}
                        </span>
                        <span className="rounded-full bg-neutral-100 px-3 py-1 text-xs font-black text-neutral-600">
                          Caja #{cash.id}
                        </span>
                      </div>

                      <div className="mt-3 text-base font-black text-neutral-950">
                        {formatDateTime(cash.openedAt)}
                      </div>
                      <div className="mt-1 text-sm font-bold text-neutral-500">
                        Cierre: {formatDateTime(cash.closedAt)}
                      </div>
                    </div>

                    <div className="grid w-full grid-cols-2 gap-3 lg:w-auto lg:min-w-[760px] lg:grid-cols-6">
                      <HistoryPaymentPill label="Ventas" value={cash.salesTotal} />
                      <HistoryPaymentPill label="Salidas totales" value={cash.movementsExpense} />
                      <HistoryPaymentPill label="Salidas efectivo" value={cash.cashMovementsExpense} />
                      <HistoryPaymentPill label="Salidas otros métodos" value={Math.max(0, Number(cash.movementsExpense || 0) - Number(cash.cashMovementsExpense || 0))} />
                      <HistoryPaymentPill label="Esperado" value={cash.closingAmountExpected} />
                      <HistoryPaymentPill label="Diferencia" value={cash.differenceAmount} />
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </ModalShell>

      {selectedCash && (
        <HistoryDetailModal
          branch={branch}
          cash={selectedCash}
          from={from}
          to={to}
          paymentMethods={paymentMethods}
          canManageFund={canManageFund}
          onClose={() => setSelectedCash(null)}
        />
      )}
    </>
  );
}


function CashAuditActivityModal({ branch, onClose }) {
  const today = new Date();
  const fromDefault = new Date();
  fromDefault.setDate(today.getDate() - 7);

  const [from, setFrom] = useState(toDateInputValue(fromDefault));
  const [to, setTo] = useState(toDateInputValue(today));
  const [actorUserId, setActorUserId] = useState('');
  const [items, setItems] = useState([]);
  const [actorOptions, setActorOptions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');

  function mergeActorOptions(logs) {
    setActorOptions((prev) => {
      const map = new Map(prev.map((actor) => [String(actor.id), actor]));
      logs.forEach((log) => {
        if (log.actorUserId) {
          map.set(String(log.actorUserId), {
            id: log.actorUserId,
            name: log.actorUserName || `Usuario #${log.actorUserId}`,
          });
        }
      });
      return Array.from(map.values()).sort((a, b) => a.name.localeCompare(b.name));
    });
  }

  async function loadActivity(nextActorUserId = actorUserId) {
    setLoading(true);
    setErrorMsg('');

    try {
      const data = await getCashAudit({
        branchId: branch.id,
        actorUserId: nextActorUserId || null,
        from,
        to,
      });
      const logs = Array.isArray(data) ? data : [];
      setItems(logs);
      mergeActorOptions(logs);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo cargar la actividad de caja.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadActivity('');
  }, []);

  function handleActorChange(value) {
    setActorUserId(value);
    loadActivity(value);
  }

  return (
    <ModalShell title="Actividad de caja" subtitle={branch?.name || 'Sede'} onClose={onClose} maxWidth="max-w-5xl">
      <div className="space-y-5">
        <div className="grid gap-3 lg:grid-cols-[1fr_1fr_1.2fr_130px]">
          <InputField label="Desde" value={from} onChange={setFrom} type="date" />
          <InputField label="Hasta" value={to} onChange={setTo} type="date" />
          <SelectField
            label="Usuario"
            value={actorUserId}
            onChange={handleActorChange}
            options={[
              { value: '', label: 'Todos los usuarios' },
              ...actorOptions.map((actor) => ({ value: String(actor.id), label: actor.name })),
            ]}
          />
          <button
            type="button"
            onClick={() => loadActivity()}
            className="self-end rounded-2xl bg-neutral-950 px-5 py-4 text-sm font-black text-white transition hover:scale-[1.01]"
          >
            Buscar
          </button>
        </div>

        <div className="rounded-[26px] border border-amber-100 bg-amber-50 p-5">
          <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-700">
            Cambios recientes
          </div>
          <div className="mt-1 text-sm font-bold text-amber-900/70">
            Auditoria de ediciones y eliminaciones en ventas, gastos, adelantos y pagos.
          </div>
        </div>

        {loading ? (
          <div className="rounded-2xl border border-neutral-200 bg-neutral-50 p-5 text-sm font-black text-neutral-500">
            Cargando actividad...
          </div>
        ) : errorMsg ? (
          <ErrorBox message={errorMsg} />
        ) : items.length === 0 ? (
          <EmptyCard title="Sin actividad" text="No hay cambios auditados para los filtros seleccionados." />
        ) : (
          <div className="space-y-3">
            {items.map((log) => (
              <div key={log.id} className="rounded-[24px] border border-neutral-200 bg-white p-4 shadow-[0_10px_28px_rgba(15,23,42,0.045)]">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <div className="text-sm font-black text-neutral-950">
                      {auditActionLabel(log.action)} {auditEntityLabel(log.entityType)} #{log.entityId || '-'}
                    </div>
                    <div className="mt-1 text-xs font-bold text-neutral-500">
                      Caja #{log.cashRegisterId || '-'} - {formatDateTime(log.createdAt)} - {log.actorUserName || 'Sistema'}
                    </div>
                    {log.reason && (
                      <div className="mt-2 rounded-xl bg-amber-50 px-3 py-2 text-xs font-bold text-amber-900">
                        Motivo: {log.reason}
                      </div>
                    )}
                  </div>

                  <details className="text-xs font-bold text-neutral-500 lg:min-w-[360px]">
                    <summary className="cursor-pointer select-none rounded-xl border border-neutral-200 bg-neutral-50 px-3 py-2 text-neutral-700">
                      Ver antes/despues
                    </summary>
                    <div className="mt-2 grid gap-2">
                      <pre className="max-h-40 overflow-auto rounded-xl bg-neutral-950 p-3 text-[11px] text-white">
                        {log.beforeSnapshot || 'Sin datos previos'}
                      </pre>
                      <pre className="max-h-40 overflow-auto rounded-xl bg-neutral-950 p-3 text-[11px] text-white">
                        {log.afterSnapshot || 'Sin datos nuevos'}
                      </pre>
                    </div>
                  </details>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </ModalShell>
  );
}

function productOrderStatusLabel(status) {
  const code = String(status || '').toUpperCase();
  if (code === 'APPROVED') return 'Aprobado';
  if (code === 'DELIVERED') return 'Entregado';
  if (code === 'REJECTED') return 'Rechazado';
  if (code === 'CANCELLED') return 'Cancelado';
  return 'Pendiente';
}

function ProductOrdersSection({ items = [], processingId, isOpen, onApprove, onReject, onDeliver }) {
  if (!items.length) return null;

  return (
    <section className="rounded-[34px] border border-blue-200 bg-blue-50 p-6 shadow-[0_16px_45px_rgba(15,23,42,0.05)]">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="text-xs font-black uppercase tracking-[0.22em] text-blue-700">
            Pedidos web
          </div>
          <h3 className="mt-2 text-2xl font-black text-neutral-950">
            Productos separados por clientes
          </h3>
          <p className="mt-1 text-sm leading-6 text-blue-900/70">
            Aprueba, rechaza o entrega pedidos del catalogo publico. Al entregar se registra una venta y se descuenta stock.
          </p>
        </div>

        <div className="rounded-2xl bg-white px-4 py-3 text-sm font-black text-blue-800">
          {items.length} pedido{items.length === 1 ? '' : 's'}
        </div>
      </div>

      <div className="mt-5 grid gap-3 xl:grid-cols-2">
        {items.map((order) => {
          const status = String(order.status || '').toUpperCase();
          const isApproved = status === 'APPROVED';
          const orderProcessingId = `product-order-${order.source || 'PRODUCT_ORDER'}-${order.id}`;
          const busy = String(processingId) === orderProcessingId;

          return (
            <article key={`${order.source || 'PRODUCT_ORDER'}-${order.id}`} className="rounded-[28px] border border-blue-100 bg-white p-4 shadow-sm">
              <div className="flex gap-3">
                <div className="flex h-16 w-16 shrink-0 items-center justify-center overflow-hidden rounded-2xl bg-blue-50 text-blue-700">
                  {order.productImageUrl ? (
                    <img src={order.productImageUrl} alt={order.productName} className="h-full w-full object-cover" />
                  ) : (
                    <span className="text-xl font-black">P</span>
                  )}
                </div>

                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="truncate text-base font-black text-neutral-950">{order.productName}</p>
                      <p className="mt-1 text-xs font-bold text-neutral-500">
                        {order.customerName} · {order.customerPhone || 'Sin telefono'}
                      </p>
                    </div>
                    <span className="rounded-full bg-blue-100 px-3 py-1 text-xs font-black text-blue-800">
                      {productOrderStatusLabel(order.status)}
                    </span>
                  </div>

                  <div className="mt-3 grid gap-2 text-xs font-black text-neutral-700 sm:grid-cols-3">
                    <div className="rounded-2xl bg-neutral-50 px-3 py-2">Cant. {order.quantity}</div>
                    <div className="rounded-2xl bg-neutral-50 px-3 py-2">{methodLabel(order.paymentMethod)}</div>
                    <div className="rounded-2xl bg-emerald-50 px-3 py-2 text-emerald-700">{formatMoney(order.total)}</div>
                  </div>

                  {order.paymentOperationNumber ? (
                    <p className="mt-2 text-xs font-bold text-neutral-500">Operacion: {order.paymentOperationNumber}</p>
                  ) : null}

                  <div className="mt-4 flex flex-wrap gap-2">
                    {status === 'PENDING' ? (
                      <>
                        <button
                          type="button"
                          onClick={() => onApprove(order)}
                          disabled={busy}
                          className="rounded-2xl bg-emerald-600 px-4 py-3 text-xs font-black text-white disabled:opacity-50"
                        >
                          Aprobar
                        </button>
                        <button
                          type="button"
                          onClick={() => onReject(order)}
                          disabled={busy}
                          className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-xs font-black text-red-700 disabled:opacity-50"
                        >
                          Rechazar
                        </button>
                      </>
                    ) : null}

                    {isApproved ? (
                      <button
                        type="button"
                        onClick={() => onDeliver(order)}
                        disabled={busy || !isOpen}
                        className="rounded-2xl bg-neutral-950 px-4 py-3 text-xs font-black text-white disabled:opacity-50"
                      >
                        Entregar y vender
                      </button>
                    ) : null}
                  </div>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function CashBranchSelector({
  branches,
  value,
  onChange,
  disabled = false,
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const selected = branches.find(
    (branch) => String(branch.id) === String(value)
  );

  useEffect(() => {
    if (!open) return undefined;

    const closeOutside = (event) => {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    };
    const closeWithKeyboard = (event) => {
      if (event.key === 'Escape') setOpen(false);
    };

    document.addEventListener('pointerdown', closeOutside);
    document.addEventListener('keydown', closeWithKeyboard);
    return () => {
      document.removeEventListener('pointerdown', closeOutside);
      document.removeEventListener('keydown', closeWithKeyboard);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative z-30 w-full min-w-[220px] sm:w-auto">
      <button
        type="button"
        disabled={disabled || branches.length === 0}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
        className="group flex w-full items-center gap-3 rounded-2xl border border-white/10 bg-white/[0.07] px-3.5 py-3 text-left shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] transition hover:border-amber-300/35 hover:bg-white/[0.11] focus:outline-none focus:ring-4 focus:ring-amber-300/15 disabled:cursor-not-allowed disabled:opacity-55 sm:min-w-[230px]"
      >
        <span className="grid h-11 w-11 shrink-0 place-items-center rounded-[15px] border border-amber-300/20 bg-amber-300/10 text-amber-300 transition group-hover:bg-amber-300/15">
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <path d="M3 10h18M5 10v10h14V10M4 4h16l1 6H3l1-6Z" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M9 20v-6h6v6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>

        <span className="min-w-0 flex-1">
          <span className="block text-[10px] font-black uppercase tracking-[0.2em] text-white/40">
            Sede operativa
          </span>
          <span className="mt-1 block truncate text-[15px] font-black text-white">
            {disabled ? 'Cargando sedes...' : selected?.name || 'Selecciona una sede'}
          </span>
        </span>

        <span className={open ? 'rotate-180 text-amber-300 transition' : 'text-white/55 transition'}>
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.4" aria-hidden="true">
            <path d="m7 10 5 5 5-5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </button>

      {open && (
        <div
          role="listbox"
          aria-label="Seleccionar sede operativa"
          className="absolute left-0 top-[calc(100%+10px)] z-50 w-full min-w-[260px] overflow-hidden rounded-[22px] border border-white/10 bg-[#17140F]/[0.98] p-2 shadow-[0_24px_70px_rgba(0,0,0,0.55)] backdrop-blur-xl sm:w-[310px]"
        >
          <div className="px-3 pb-2 pt-1 text-[10px] font-black uppercase tracking-[0.2em] text-white/35">
            Cambiar sede
          </div>
          <div className="max-h-64 space-y-1 overflow-y-auto">
            {branches.map((branch) => {
              const active = String(branch.id) === String(value);
              return (
                <button
                  key={branch.id}
                  type="button"
                  role="option"
                  aria-selected={active}
                  onClick={() => {
                    onChange(String(branch.id));
                    setOpen(false);
                  }}
                  className={
                    active
                      ? 'flex w-full items-center gap-3 rounded-2xl border border-amber-300/25 bg-amber-300/10 px-3 py-3 text-left text-white'
                      : 'flex w-full items-center gap-3 rounded-2xl border border-transparent px-3 py-3 text-left text-white/75 transition hover:border-white/10 hover:bg-white/[0.07] hover:text-white'
                  }
                >
                  <span className={active ? 'grid h-9 w-9 place-items-center rounded-xl bg-amber-300 text-sm font-black text-neutral-950' : 'grid h-9 w-9 place-items-center rounded-xl bg-white/[0.07] text-sm font-black text-white/70'}>
                    {String(branch.name || 'S').trim().charAt(0).toUpperCase()}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-sm font-black">
                    {branch.name}
                  </span>
                  {active && (
                    <span className="grid h-7 w-7 place-items-center rounded-full bg-emerald-400/15 text-emerald-300">
                      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2.7" aria-hidden="true">
                        <path d="m5 12 4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
function CashOverviewCard({ eyebrow, title, primaryLabel, primaryValue, secondaryLabel, secondaryValue, tone = 'light' }) {
  const dark = tone === 'dark';
  const gold = tone === 'gold';
  return (
    <article className={
      dark
        ? 'rounded-[26px] border border-neutral-900 bg-neutral-950 p-5 text-white shadow-[0_18px_40px_rgba(15,23,42,0.14)]'
        : gold
          ? 'rounded-[26px] border border-amber-300 bg-amber-50/50 p-5 text-neutral-950 shadow-sm'
          : 'rounded-[26px] border border-neutral-200 bg-white p-5 text-neutral-950 shadow-sm'
    }>
      <div className={dark ? 'text-[10px] font-black uppercase tracking-[0.2em] text-white/40' : 'text-[10px] font-black uppercase tracking-[0.2em] text-amber-700'}>
        {eyebrow}
      </div>
      <h3 className="mt-2 text-base font-black">{title}</h3>
      <div className={dark ? 'mt-4 grid grid-cols-2 divide-x divide-white/10' : 'mt-4 grid grid-cols-2 divide-x divide-neutral-200'}>
        <div className="pr-4">
          <div className={dark ? 'text-xs font-bold text-white/45' : 'text-xs font-bold text-neutral-500'}>{primaryLabel}</div>
          <div className="mt-1 text-2xl font-black">{primaryValue}</div>
        </div>
        <div className="pl-4">
          <div className={dark ? 'text-xs font-bold text-white/45' : 'text-xs font-bold text-neutral-500'}>{secondaryLabel}</div>
          <div className="mt-1 text-2xl font-black">{secondaryValue}</div>
        </div>
      </div>
    </article>
  );
}
export default function OwnerCashPage() {
  const { session } = useAuth();
  const [branches, setBranches] = useState([]);
  const [selectedBranchId, setSelectedBranchId] = useState('');
  const [cashRegister, setCashRegister] = useState(null);
  const [pendingReconciliation, setPendingReconciliation] = useState(null);
  const [movements, setMovements] = useState([]);
  const [sales, setSales] = useState([]);
  const [pendingValidationSales, setPendingValidationSales] = useState([]);
  const [productOrders, setProductOrders] = useState([]);
  const [paymentMethods, setPaymentMethods] = useState(DEFAULT_PAYMENT_METHODS);

  const [loadingBranches, setLoadingBranches] = useState(true);
  const [loadingCash, setLoadingCash] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const [showOpenModal, setShowOpenModal] = useState(false);
  const [showReconciliationModal, setShowReconciliationModal] = useState(false);
  const [showCloseModal, setShowCloseModal] = useState(false);
  const [showMovementModal, setShowMovementModal] = useState(false);
  const [showFundMovementModal, setShowFundMovementModal] = useState(false);
  const [showSaleModal, setShowSaleModal] = useState(false);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [showAuditActivityModal, setShowAuditActivityModal] = useState(false);
  const [showBarberPaymentModal, setShowBarberPaymentModal] = useState(false);
  const [editingMovement, setEditingMovement] = useState(null);
  const [editingSale, setEditingSale] = useState(null);
  const [viewingSale, setViewingSale] = useState(null);
  const [fiscalRefreshKey, setFiscalRefreshKey] = useState(0);
  const [pendingAppointment, setPendingAppointment] = useState(null);
  const [showAppointmentSaleModal, setShowAppointmentSaleModal] = useState(false);
  const [lastUpdatedAt, setLastUpdatedAt] = useState(null);
  const [processingApprovalId, setProcessingApprovalId] = useState(null);
  const [permissionBundle, setPermissionBundle] = useState(null);

  const currentRole = String(session?.role || '').trim().toUpperCase();
  const canManageSales =
    currentRole === 'OWNER' ||
    hasAnyOwnerPermission(permissionBundle, ['CASH_ACCESS']);
  const canManageFund =
    currentRole === 'OWNER' ||
    (currentRole === 'ADMIN' &&
      hasAnyOwnerPermission(permissionBundle, ['CASH_FUND_MANAGE']));
  const labels = useMemo(
    () => getBusinessLabels(session?.businessType),
    [session?.businessType]
  );

  const selectedBranch = useMemo(() => {
    return branches.find((item) => String(item.id) === String(selectedBranchId)) || null;
  }, [branches, selectedBranchId]);

  useEffect(() => {
    let mounted = true;

    async function loadPermissions() {
      try {
        const data = await getMyOwnerPermissions();
        if (mounted) setPermissionBundle(data);
      } catch {
        if (mounted) {
          setPermissionBundle({
            owner: currentRole === 'OWNER',
            permissions: [],
          });
        }
      }
    }

    if (session?.token) {
      loadPermissions();
    } else {
      setPermissionBundle(null);
    }

    return () => {
      mounted = false;
    };
  }, [session?.token, currentRole]);

  async function loadBranches() {
    setLoadingBranches(true);
    setErrorMsg('');

    try {
      const data = await getOwnerBranches();
      setBranches(data);

      if (data.length > 0) {
        setSelectedBranchId((prev) => prev || String(data[0].id));
      }
    } catch (error) {
      setErrorMsg(error.message || 'No se pudieron cargar las sedes.');
    } finally {
      setLoadingBranches(false);
    }
  }

  async function loadPaymentMethods(branchId = selectedBranchId) {
    if (!branchId) {
        setPaymentMethods(DEFAULT_PAYMENT_METHODS);
      return;
    }

    try {
      const data = await getOwnerPaymentMethods(branchId);
      setPaymentMethods(normalizePaymentMethodOptions(data));
    } catch {
        setPaymentMethods(DEFAULT_PAYMENT_METHODS);
    }
  }

  async function loadCash(branchId = selectedBranchId, { silent = false } = {}) {
    if (!branchId) return;

    if (silent) {
      setRefreshing(true);
    } else {
      setLoadingCash(true);
    }

    setErrorMsg('');

    try {
      const [current, pendingCash] = await Promise.all([
        getCurrentCashRegister(branchId),
        getPendingCashReconciliation(branchId),
      ]);
      setCashRegister(current);
      setPendingReconciliation(pendingCash);

      let dataPendingSales = [];
      let dataProductOrders = [];
      try {
        dataPendingSales = await getPendingValidationSales(branchId);
      } catch {
        dataPendingSales = [];
      }
      try {
        const [pendingOrders, approvedOrders] = await Promise.all([
          getOwnerProductOrders({ branchId, status: 'PENDING' }),
          getOwnerProductOrders({ branchId, status: 'APPROVED' }),
        ]);
        dataProductOrders = [...pendingOrders, ...approvedOrders];
      } catch {
        dataProductOrders = [];
      }
      setPendingValidationSales(Array.isArray(dataPendingSales) ? dataPendingSales : []);
      setProductOrders(Array.isArray(dataProductOrders) ? dataProductOrders : []);

      if (current?.id) {
        const [dataMovements, dataSales] = await Promise.all([
          getCashMovements({
            branchId,
            cashRegisterId: current.id,
          }),
          getTodayCashSales(branchId),
        ]);

        setMovements(dataMovements);
        setSales(Array.isArray(dataSales) ? dataSales : []);
      } else {
        setMovements([]);
        setSales([]);
      }

      setLastUpdatedAt(new Date());
    } catch (error) {
      setPendingValidationSales([]);
      setProductOrders([]);
      setErrorMsg(error.message || 'No se pudo cargar la caja.');
    } finally {
      setLoadingCash(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    loadBranches();

    const pending = readAttendAppointmentFromStorage();
    if (pending) {
      setPendingAppointment(pending);
      if (pending.branchId) {
        setSelectedBranchId(String(pending.branchId));
      }
    }
  }, []);

  useEffect(() => {
    if (selectedBranchId) {
      loadPaymentMethods(selectedBranchId);
      loadCash(selectedBranchId);
    }
  }, [selectedBranchId]);

  useEffect(() => {
    if (!selectedBranchId) return undefined;

    const timer = window.setInterval(() => {
      if (
        showOpenModal ||
        showCloseModal ||
        showMovementModal ||
        showFundMovementModal ||
        showSaleModal ||
        showHistoryModal ||
        showAuditActivityModal ||
        showBarberPaymentModal ||
        editingMovement ||
        editingSale ||
        viewingSale ||
        showAppointmentSaleModal ||
        processingApprovalId
      ) {
        return;
      }

      loadCash(selectedBranchId, { silent: true });
    }, 10000);

    return () => window.clearInterval(timer);
  }, [
    selectedBranchId,
    showOpenModal,
    showCloseModal,
    showMovementModal,
    showFundMovementModal,
    showSaleModal,
    showHistoryModal,
    showAuditActivityModal,
    showBarberPaymentModal,
    editingMovement,
    editingSale,
    viewingSale,
    showAppointmentSaleModal,
    processingApprovalId,
  ]);

  
  function dismissPendingAppointment() {
    clearAttendAppointmentFromStorage();
    setPendingAppointment(null);
    setShowAppointmentSaleModal(false);
  }

  async function handleDeleteSale(sale) {
    if (!canManageSales) return;

    const saleId = saleIdOf(sale);
    if (!selectedBranch || !saleId) return;

    const auditReason = window.prompt('Motivo para eliminar esta venta:');
    if (auditReason === null) return;
    if (!auditReason.trim()) {
      setErrorMsg('El motivo de auditoria es obligatorio para eliminar una venta.');
      return;
    }
    const ok = await premiumConfirm('¿Seguro que deseas eliminar esta venta? Esta acción no se puede deshacer.');
    if (!ok) return;

    setErrorMsg('');

    try {
      await deleteCashSale({
        branchId: selectedBranch.id,
        saleId,
        auditReason: auditReason.trim(),
      });

      await loadCash(selectedBranchId);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo eliminar la venta.');
    }
  }

  async function handleDeleteMovement(movement) {
    if (!selectedBranch || !movement?.id) return;
    if (
      String(movement?.fundingSource || '').toUpperCase() ===
        'ACCUMULATED_FUND' &&
      !canManageFund
    ) {
      setErrorMsg('No tienes permiso para eliminar movimientos del fondo acumulado.');
      return;
    }

    const auditReason = window.prompt('Motivo para eliminar este movimiento:');
    if (auditReason === null) return;
    if (!auditReason.trim()) {
      setErrorMsg('El motivo de auditoria es obligatorio para eliminar un movimiento.');
      return;
    }
    const ok = await premiumConfirm('¿Seguro que deseas eliminar este movimiento? Esta acción no se puede deshacer.');
    if (!ok) return;

    setErrorMsg('');

    try {
      await deleteCashMovement({
        branchId: selectedBranch.id,
        movementId: movement.id,
        auditReason: auditReason.trim(),
      });

      await loadCash(selectedBranchId);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo eliminar el movimiento.');
    }
  }

  async function handleApproveMovement(movement) {
    if (!selectedBranch || !movement?.id) return;

    const ok = await premiumConfirm(
      `¿Aprobar el movimiento "${movement.concept || 'Movimiento'}" por ${formatMoney(movement.amount)}?`
    );
    if (!ok) return;

    setErrorMsg('');
    setProcessingApprovalId(movement.id);

    try {
      await approveCashMovement({
        branchId: selectedBranch.id,
        movementId: movement.id,
        note: 'Ingreso aprobado desde caja web',
      });

      await loadCash(selectedBranchId);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo aprobar el movimiento pendiente.');
    } finally {
      setProcessingApprovalId(null);
    }
  }

  async function handleRejectMovement(movement) {
    if (!selectedBranch || !movement?.id) return;

    const ok = await premiumConfirm(
      `¿Rechazar el movimiento "${movement.concept || 'Movimiento'}" por ${formatMoney(movement.amount)}?`
    );
    if (!ok) return;

    setErrorMsg('');
    setProcessingApprovalId(movement.id);

    try {
      await rejectCashMovement({
        branchId: selectedBranch.id,
        movementId: movement.id,
        note: 'Ingreso rechazado desde caja web',
      });

      await loadCash(selectedBranchId);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo rechazar el movimiento pendiente.');
    } finally {
      setProcessingApprovalId(null);
    }
  }

  async function handleApproveSalePayment(sale) {
    if (!selectedBranch) return;

    const saleId = saleIdOf(sale);
    if (!saleId) return;

    const ok = await premiumConfirm(
      `Aprobar la venta #${saleId} por ${formatMoney(sale.total)}?`
    );
    if (!ok) return;

    let pendingElectronicDocument = null;
    try {
      const stored = window.sessionStorage.getItem(`gods.pending-electronic.${saleId}`);
      pendingElectronicDocument = stored ? JSON.parse(stored) : null;
    } catch {
      pendingElectronicDocument = null;
    }
    const preparedPrintPopup = pendingElectronicDocument?.printAfterIssue
      ? window.open('', '_blank')
      : null;
    if (preparedPrintPopup) {
      preparedPrintPopup.document.write('<title>Preparando comprobante</title><p style="font-family:system-ui;padding:32px">Aprobando la venta y preparando el comprobante oficial...</p>');
    }

    setErrorMsg('');
    setProcessingApprovalId(`sale-${saleId}`);

    try {
      const approvedSale = await approveSalePayment({
        branchId: selectedBranch.id,
        saleId,
      });

      if (pendingElectronicDocument) {
        try {
          const electronicDocument = await issueElectronicDocument({
            saleId,
            branchId: selectedBranch.id,
            documentType: pendingElectronicDocument.documentType,
            receiverDocumentType: pendingElectronicDocument.documentType === 'INVOICE' ? '6' : '1',
            receiverDocumentNumber: pendingElectronicDocument.receiverDocumentNumber,
            receiverName: pendingElectronicDocument.receiverName,
            receiverAddress: pendingElectronicDocument.receiverAddress || '-',
            receiverEmail: pendingElectronicDocument.receiverEmail || null,
          });
          window.sessionStorage.removeItem(`gods.pending-electronic.${saleId}`);
          const accepted = electronicDocument?.status === 'ACCEPTED' || electronicDocument?.status === 'ACCEPTED_WITH_OBSERVATIONS';
          if (accepted && pendingElectronicDocument.printAfterIssue) {
            await openElectronicPdfForPrint(electronicDocument, preparedPrintPopup);
          } else {
            preparedPrintPopup?.close();
            window.alert(`Venta aprobada. El comprobante quedó en estado ${electronicDocument?.status || 'PENDIENTE'}; puedes actualizarlo desde Ver detalle.`);
          }
        } catch (reason) {
          preparedPrintPopup?.close();
          const electronicDocumentError = reason.message || 'No se pudo emitir el comprobante después de aprobar la venta.';
          setViewingSale({
            ...approvedSale,
            electronicDocumentError,
            pendingElectronicDocument,
          });
          window.alert(`La venta fue aprobada, pero el comprobante no pudo emitirse: ${electronicDocumentError}`);
        }
      } else {
        const printTicket = window.confirm('Venta aprobada correctamente.\n\n¿Deseas imprimir el ticket convencional ahora?');
        if (printTicket) {
          void autoPrintApprovedSale(approvedSale, { branchId: selectedBranch.id });
        }
      }

      offerCustomerWhatsappFollowUp(saleWithWhatsappFallback(approvedSale, sale), {
        canOpenWhatsapp: currentRole === 'OWNER',
      });
      await loadCash(selectedBranchId);
      setFiscalRefreshKey((value) => value + 1);
    } catch (error) {
      preparedPrintPopup?.close();
      setErrorMsg(error.message || 'No se pudo aprobar la venta pendiente.');
    } finally {
      setProcessingApprovalId(null);
    }
  }

  async function handleRejectSalePayment(sale) {
    if (!selectedBranch) return;

    const saleId = saleIdOf(sale);
    if (!saleId) return;

    const reason = window.prompt(
      `Motivo para rechazar la venta #${saleId}:`,
      'Pago no validado'
    );
    if (reason === null) return;

    const cleanReason = reason.trim() || 'Pago rechazado desde caja web';

    setErrorMsg('');
    setProcessingApprovalId(`sale-${saleId}`);

    try {
      await rejectSalePayment({
        branchId: selectedBranch.id,
        saleId,
        reason: cleanReason,
      });

      await loadCash(selectedBranchId);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo rechazar la venta pendiente.');
    } finally {
      setProcessingApprovalId(null);
    }
  }

  async function handleApproveProductOrder(order) {
    if (!selectedBranch || !order?.id) return;

    setErrorMsg('');
    setProcessingApprovalId(`product-order-${order.source || 'PRODUCT_ORDER'}-${order.id}`);

    try {
      await approveOwnerProductOrder({
        branchId: selectedBranch.id,
        orderId: order.id,
        source: order.source,
        note: 'Pedido aprobado desde caja web',
      });
      await loadCash(selectedBranchId);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo aprobar el pedido de producto.');
    } finally {
      setProcessingApprovalId(null);
    }
  }

  async function handleRejectProductOrder(order) {
    if (!selectedBranch || !order?.id) return;

    const reason = window.prompt(
      `Motivo para rechazar el pedido de ${order.productName}:`,
      'Producto no disponible'
    );
    if (reason === null) return;

    setErrorMsg('');
    setProcessingApprovalId(`product-order-${order.source || 'PRODUCT_ORDER'}-${order.id}`);

    try {
      await rejectOwnerProductOrder({
        branchId: selectedBranch.id,
        orderId: order.id,
        source: order.source,
        note: reason.trim() || 'Pedido rechazado desde caja web',
      });
      await loadCash(selectedBranchId);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo rechazar el pedido de producto.');
    } finally {
      setProcessingApprovalId(null);
    }
  }

  async function handleDeliverProductOrder(order) {
    if (!selectedBranch || !order?.id) return;

    if (!isOpen) {
      setErrorMsg('Abre caja antes de entregar un pedido.');
      return;
    }

    const ok = await premiumConfirm(
      `Entregar ${order.quantity} x ${order.productName} por ${formatMoney(order.total)}?`
    );
    if (!ok) return;

    setErrorMsg('');
    setProcessingApprovalId(`product-order-${order.source || 'PRODUCT_ORDER'}-${order.id}`);

    try {
      await deliverOwnerProductOrder({
        branchId: selectedBranch.id,
        orderId: order.id,
        source: order.source,
        note: 'Pedido entregado desde caja web',
      });
      await loadCash(selectedBranchId);
    } catch (error) {
      setErrorMsg(error.message || 'No se pudo entregar el pedido de producto.');
    } finally {
      setProcessingApprovalId(null);
    }
  }

  const isOpen = String(cashRegister?.status || '').toUpperCase() === 'OPEN';

  const paymentSalesSource = Array.isArray(cashRegister?.paymentMethodsSummary)
    ? cashRegister.paymentMethodsSummary
    : [];

  const paymentBalanceSource = Array.isArray(cashRegister?.paymentMethodBalances)
    ? cashRegister.paymentMethodBalances
    : [];

  const paymentRows = buildPaymentMethodRows(paymentSalesSource, paymentBalanceSource, paymentMethods);
  const cashRow = paymentRows.find((item) => item.code === 'CASH');
  const digitalBalance = paymentRows
    .filter((item) => item.code !== 'CASH')
    .reduce((sum, item) => sum + Number(item.balanceAmount || 0), 0);
  const digitalSales = paymentRows
    .filter((item) => item.code !== 'CASH')
    .reduce((sum, item) => sum + Number(item.salesAmount || 0), 0);

  const expected = Number(cashRegister?.closingAmountExpected || 0);
  const accumulatedFundBalance = Number(cashRegister?.accumulatedFundBalance || cashRegister?.fundBalance || 0);
  const openingAmount = Number(cashRegister?.openingAmount || 0);
  const salesTotal = Number(cashRegister?.salesTotal || 0);
  const cashSalesTotal = Number(cashRow?.salesAmount ?? cashRegister?.cashSalesTotal ?? 0);
  const cashBalance = Number(cashRow?.balanceAmount ?? expected);
  const income = Number(cashRegister?.movementsIncome || 0);
  const derivedCashExpense = movements
    .filter((movement) =>
      ['EXPENSE', 'ADVANCE_BARBER', 'PAYMENT_BARBER'].includes(String(movement?.type || '').toUpperCase())
    )
    .filter((movement) => normalizeMethod(movement?.paymentMethod) === 'CASH')
    .filter((movement) => String(movement?.fundingSource || 'CASH_REGISTER').toUpperCase() === 'CASH_REGISTER')
    .reduce((sum, movement) => sum + Number(movement?.amount || 0), 0);
  const cashExpense = Number(cashRegister?.cashMovementsExpense ?? derivedCashExpense);
  const courtesySummary = buildCourtesySummary(sales);
  const pendingSales = pendingValidationSales.filter(isPendingSaleValidation);
  const pendingApprovalMovements = movements.filter(isPendingOwnerApproval);
  const visibleMovements = movements.filter(
    (movement) =>
      !isPendingOwnerApproval(movement) &&
      (canManageFund ||
        String(movement?.fundingSource || '').toUpperCase() !==
          'ACCUMULATED_FUND')
  );

  return (
    <div className="space-y-7">
      {pendingReconciliation && (
        <section className="rounded-[28px] border border-amber-300 bg-amber-50 p-5 shadow-sm">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="text-xs font-black uppercase tracking-[0.18em] text-amber-700">Conciliacion pendiente</div>
              <h3 className="mt-2 text-xl font-black text-neutral-950">Una caja anterior se cerro automaticamente</h3>
              <p className="mt-2 text-sm font-semibold text-neutral-600">
                {canManageFund ? (
                  <>Esperado {formatMoney(pendingReconciliation.closingAmountExpected)}. Confirma el efectivo, registra un gasto olvidado o envía el saldo al fondo acumulado.</>
                ) : (
                  <>Esperado {formatMoney(pendingReconciliation.closingAmountExpected)}. Confirma el efectivo y registra cualquier gasto olvidado.</>
                )}
              </p>
            </div>
            <button type="button" onClick={() => setShowReconciliationModal(true)} className="rounded-2xl bg-neutral-950 px-5 py-3 text-sm font-black text-white">
              Revisar cierre
            </button>
          </div>
        </section>
      )}
      <section className="relative rounded-[34px] border border-amber-400/15 bg-[linear-gradient(135deg,#090909_0%,#15110A_42%,#101827_100%)] p-5 text-white shadow-[0_22px_60px_rgba(15,23,42,0.18)] sm:p-6">
        <div className="pointer-events-none absolute inset-0 overflow-hidden rounded-[34px]">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_18%_0%,rgba(251,191,36,0.18),transparent_30%),radial-gradient(circle_at_100%_0%,rgba(59,130,246,0.14),transparent_32%)]" />
        </div>

        <div className="relative">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <div className="text-[11px] font-black uppercase tracking-[0.22em] text-amber-300/80">
                Caja web
              </div>
              <h2 className="mt-2 text-3xl font-black tracking-tight sm:text-4xl">
                Control de caja
              </h2>
              <p className="mt-2 max-w-xl text-sm leading-6 text-white/55">
                Operación diaria, saldos y movimientos de la sede seleccionada.
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => loadCash(selectedBranchId, { silent: true })}
                disabled={refreshing}
                className="rounded-xl border border-white/10 bg-white/[0.07] px-3.5 py-2.5 text-xs font-black text-white/80 transition hover:bg-white/[0.12] disabled:opacity-60"
              >
                {refreshing ? 'Actualizando...' : '↻ Actualizar'}
              </button>
              <button
                onClick={() => setShowHistoryModal(true)}
                disabled={!selectedBranch}
                className="rounded-xl border border-white/10 bg-white/[0.07] px-3.5 py-2.5 text-xs font-black text-white/80 transition hover:bg-white/[0.12] disabled:opacity-60"
              >
                Historial
              </button>
              <button
                onClick={() => setShowAuditActivityModal(true)}
                disabled={!selectedBranch}
                className="rounded-xl border border-white/10 bg-white/[0.07] px-3.5 py-2.5 text-xs font-black text-white/80 transition hover:bg-white/[0.12] disabled:opacity-60"
              >
                Actividad
              </button>
              {canManageFund && (
                <button
                  onClick={() => setShowFundMovementModal(true)}
                  disabled={!selectedBranch}
                  className="rounded-xl border border-amber-300/25 bg-amber-300/10 px-3.5 py-2.5 text-xs font-black text-amber-200 transition hover:bg-amber-300/15 disabled:opacity-50"
                >
                  Fondo acumulado
                </button>
              )}
            </div>
          </div>

          <div className="mt-6 grid gap-4 xl:grid-cols-[minmax(300px,0.8fr)_minmax(520px,1.2fr)]">
            <div className="rounded-[24px] border border-white/10 bg-white/[0.055] p-3.5">
              <div className="flex flex-col gap-3 sm:flex-row">
                <CashBranchSelector
                  branches={branches}
                  value={selectedBranchId}
                  onChange={setSelectedBranchId}
                  disabled={loadingBranches}
                />
                <div className={
                  isOpen
                    ? 'flex min-w-[150px] flex-1 items-center gap-3 rounded-2xl border border-emerald-400/20 bg-emerald-400/10 px-4 py-3'
                    : 'flex min-w-[150px] flex-1 items-center gap-3 rounded-2xl border border-red-400/20 bg-red-400/10 px-4 py-3'
                }>
                  <span className={isOpen ? 'h-2.5 w-2.5 rounded-full bg-emerald-300 shadow-[0_0_14px_rgba(110,231,183,0.8)]' : 'h-2.5 w-2.5 rounded-full bg-red-300'} />
                  <div>
                    <div className={isOpen ? 'text-[10px] font-black uppercase tracking-[0.16em] text-emerald-300/65' : 'text-[10px] font-black uppercase tracking-[0.16em] text-red-300/70'}>
                      Estado
                    </div>
                    <div className={isOpen ? 'mt-0.5 text-sm font-black text-emerald-300' : 'mt-0.5 text-sm font-black text-red-200'}>
                      {isOpen ? 'Caja abierta' : 'Caja cerrada'}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 px-1 text-xs font-bold text-white/45">
                {cashRegister?.openedAt && (
                  <span><strong className="text-white/70">Apertura:</strong> {formatDateTime(cashRegister.openedAt)}</span>
                )}
                {lastUpdatedAt && (
                  <span>
                    <strong className="text-white/70">Actualizado:</strong>{' '}
                    {lastUpdatedAt.toLocaleTimeString('es-PE', { hour: '2-digit', minute: '2-digit' })}
                  </span>
                )}
              </div>
            </div>

            <div className="rounded-[24px] border border-white/10 bg-black/15 p-3.5">
              <div className="mb-3 text-[10px] font-black uppercase tracking-[0.2em] text-white/35">
                Operaciones
              </div>
              {isOpen ? (
                <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
                  <button
                    onClick={() => setShowSaleModal(true)}
                    disabled={!canManageSales}
                    className="rounded-2xl bg-emerald-400 px-4 py-3.5 text-sm font-black text-neutral-950 transition hover:bg-emerald-300 disabled:opacity-50"
                  >
                    Nueva venta
                  </button>
                  <button
                    onClick={() => setShowMovementModal(true)}
                    className="rounded-2xl bg-amber-400 px-4 py-3.5 text-sm font-black text-neutral-950 transition hover:bg-amber-300"
                  >
                    Movimiento
                  </button>
                  <button
                    onClick={() => setShowBarberPaymentModal(true)}
                    className="rounded-2xl border border-white/15 bg-white/10 px-4 py-3.5 text-sm font-black text-white transition hover:bg-white/15"
                  >
                    Pagar {labels.professionalSingular}
                  </button>
                  <button
                    onClick={() => setShowCloseModal(true)}
                    className="rounded-2xl border border-red-300/25 bg-red-300/10 px-4 py-3.5 text-sm font-black text-red-200 transition hover:bg-red-300/15"
                  >
                    Cerrar caja
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setShowOpenModal(true)}
                  disabled={!selectedBranch}
                  className="w-full rounded-2xl bg-amber-400 px-5 py-3.5 text-sm font-black text-neutral-950 transition hover:bg-amber-300 disabled:opacity-60"
                >
                  Abrir caja
                </button>
              )}
            </div>
          </div>
        </div>
      </section>
      {errorMsg && <ErrorBox message={errorMsg} />}

      {pendingAppointment && (
        <AppointmentSaleBanner
          appointment={pendingAppointment}
          isOpen={isOpen}
          onOpenSale={() => setShowAppointmentSaleModal(true)}
          onDismiss={dismissPendingAppointment}
        />
      )}

      {loadingBranches || loadingCash ? (
        <div className="rounded-[28px] border border-neutral-200 bg-white p-6 font-bold text-neutral-700 shadow-sm">
          Cargando caja...
        </div>
      ) : !selectedBranch ? (
        <EmptyCard
          title="No hay sedes disponibles"
          text="No se encontraron sedes conectadas al usuario actual."
        />
      ) : !cashRegister ? (
        <>
          <PendingSaleValidationsSection
            items={pendingSales}
            processingId={processingApprovalId}
            labels={labels}
            onView={(sale) => setViewingSale(sale)}
            onApprove={handleApproveSalePayment}
            onReject={handleRejectSalePayment}
          />

          <ProductOrdersSection
            items={productOrders}
            processingId={processingApprovalId}
            isOpen={isOpen}
            onApprove={handleApproveProductOrder}
            onReject={handleRejectProductOrder}
            onDeliver={handleDeliverProductOrder}
          />

          <EmptyCard
        title="No hay caja abierta"
        text="Abre una caja para empezar a registrar ventas, ingresos, gastos y movimientos del día."
        action={
          <button
            onClick={() => setShowOpenModal(true)}
            disabled={!selectedBranch}
            className="rounded-2xl bg-amber-400 px-6 py-4 text-sm font-black text-neutral-950 shadow-[0_16px_35px_rgba(251,191,36,0.22)] transition hover:scale-[1.02] disabled:opacity-60"
          >
            Abrir caja ahora
          </button>
        }
          />
        </>
      ) : (
        <>
          <section className="grid gap-4 lg:grid-cols-3">
            <CashOverviewCard
              eyebrow="Disponibilidad"
              title="Fondos disponibles"
              primaryLabel="Fondo acumulado"
              primaryValue={canManageFund ? formatMoney(accumulatedFundBalance) : 'Restringido'}
              secondaryLabel="Digital"
              secondaryValue={formatMoney(digitalBalance)}
              tone="gold"
            />
            <CashOverviewCard
              eyebrow="Operación"
              title="Movimiento del día"
              primaryLabel="Apertura"
              primaryValue={formatMoney(openingAmount)}
              secondaryLabel="Ventas"
              secondaryValue={formatMoney(salesTotal)}
            />
            <CashOverviewCard
              eyebrow="Control"
              title="Efectivo en caja"
              primaryLabel="Físico"
              primaryValue={formatMoney(cashBalance)}
              secondaryLabel="Esperado"
              secondaryValue={formatMoney(expected)}
              tone="dark"
            />
          </section>

          <CourtesySummarySection summary={courtesySummary} labels={labels} />

          <PendingSaleValidationsSection
            items={pendingSales}
            processingId={processingApprovalId}
            labels={labels}
            onView={(sale) => setViewingSale(sale)}
            onApprove={handleApproveSalePayment}
            onReject={handleRejectSalePayment}
          />

          <PendingCashApprovalsSection
            items={pendingApprovalMovements}
            processingId={processingApprovalId}
            onApprove={handleApproveMovement}
            onReject={handleRejectMovement}
          />

          <ProductOrdersSection
            items={productOrders}
            processingId={processingApprovalId}
            isOpen={isOpen}
            onApprove={handleApproveProductOrder}
            onReject={handleRejectProductOrder}
            onDeliver={handleDeliverProductOrder}
          />

          <section className="rounded-[34px] border border-neutral-200 bg-white p-6 shadow-[0_16px_45px_rgba(15,23,42,0.05)]">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-600">
                  Conciliación por método
                </div>
                <h3 className="mt-2 text-2xl font-black text-neutral-950">
                  Dinero esperado por método de pago
                </h3>
                <p className="mt-1 max-w-3xl text-sm leading-6 text-neutral-500">
                  El efectivo incluye apertura, ventas en efectivo, ingresos, gastos, pagos a {labels.professionalsPlural} y traslados.
                  Los métodos digitales muestran el saldo disponible después de movimientos.
                </p>
              </div>

              <div className="grid gap-2 text-sm sm:grid-cols-2">
                <div className="rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3">
                  <div className="text-xs font-black uppercase tracking-[0.16em] text-emerald-700">Efectivo vendido</div>
                  <div className="mt-1 text-lg font-black text-emerald-800">{formatMoney(cashSalesTotal)}</div>
                </div>
                <div className="rounded-2xl border border-blue-200 bg-blue-50 px-4 py-3">
                  <div className="text-xs font-black uppercase tracking-[0.16em] text-blue-700">Digital vendido</div>
                  <div className="mt-1 text-lg font-black text-blue-800">{formatMoney(digitalSales)}</div>
                </div>
              </div>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-5">
              {paymentRows.map((method) => (
                <PaymentMethodCard key={method.code} method={method} />
              ))}
            </div>
          </section>

<CashNegativeAlert
  expected={expected}
  cashSalesTotal={cashSalesTotal}
  expense={cashExpense}
/>

<section className="grid gap-5 xl:grid-cols-[0.78fr_1.22fr]">
            <div className="rounded-[32px] border border-neutral-200 bg-white p-6 shadow-[0_16px_45px_rgba(15,23,42,0.05)]">
              <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-600">
                Caja física
              </div>

              <h3 className="mt-2 text-2xl font-black text-neutral-950">
                Fórmula del efectivo esperado
              </h3>

              <p className="mt-1 text-sm leading-6 text-neutral-500">
                Este cálculo te dice cuánto dinero físico deberías tener en la caja.
              </p>

              <div className="mt-5 space-y-3">
                <div className="rounded-[24px] border border-emerald-200 bg-emerald-50 p-4">
                  <div className="text-xs font-black uppercase tracking-[0.18em] text-emerald-700">
                    Entrada de efectivo
                  </div>
                  <div className="mt-2 text-sm font-black text-neutral-950">
                    Apertura {formatMoney(openingAmount)} + ventas efectivo {formatMoney(cashSalesTotal)} + ingresos {formatMoney(income)}
                  </div>
                </div>

                <div className="rounded-[24px] border border-red-100 bg-red-50 p-4">
                  <div className="text-xs font-black uppercase tracking-[0.18em] text-red-700">
                    Salidas de efectivo
                  </div>
                  <div className="mt-2 text-2xl font-black text-red-700">
                    {formatMoney(cashExpense)}
                  </div>
                  <div className="mt-1 text-xs text-red-500">
                    Gastos, adelantos, pagos de {labels.professionalSingular} y salidas registradas.
                  </div>
                </div>

                <div className="rounded-[24px] border border-amber-200 bg-amber-50 p-4">
                  <div className="text-xs font-black uppercase tracking-[0.18em] text-amber-700">
                    Resultado esperado
                  </div>
                  <div className="mt-2 text-xl font-black text-neutral-950">
                    {formatMoney(expected)}
                  </div>
                  <div className="mt-1 text-xs text-neutral-500">
                    Si el monto físico contado no coincide, revisa movimientos o pagos registrados con método incorrecto.
                  </div>
                </div>
              </div>
            </div>

            <div className="rounded-[32px] border border-neutral-200 bg-white p-6 shadow-[0_16px_45px_rgba(15,23,42,0.05)]">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <div className="text-xs font-black uppercase tracking-[0.22em] text-amber-600">
                    Movimientos
                  </div>

                  <h3 className="mt-2 text-2xl font-black text-neutral-950">
                    Actividad de caja
                  </h3>

                  <p className="mt-1 text-sm text-neutral-500">
                    Ingresos, gastos y traslados registrados en la caja abierta.
                  </p>
                </div>

                <div className="rounded-2xl bg-neutral-100 px-4 py-3 text-sm font-black text-neutral-700">
                  {visibleMovements.length} movimiento{visibleMovements.length === 1 ? '' : 's'}
                </div>
              </div>

              <div className="mt-6 overflow-hidden rounded-[26px] border border-neutral-200">
                <table className="w-full text-left text-sm">
                  <thead className="bg-[linear-gradient(135deg,#090909_0%,#111827_100%)] text-white">
                    <tr>
                      <th className="px-5 py-4 font-black">Tipo</th>
                      <th className="px-5 py-4 font-black">Concepto</th>
                      <th className="px-5 py-4 font-black">Método</th>
                      <th className="px-5 py-4 font-black">Monto</th>
                      <th className="px-5 py-4 font-black">Fecha</th>
                      <th className="px-5 py-4 font-black text-right">Acciones</th>
                    </tr>
                  </thead>

                  <tbody>
                    {visibleMovements.length === 0 ? (
                      <tr>
                        <td className="px-5 py-6 text-neutral-500" colSpan="6">
                          No hay movimientos registrados.
                        </td>
                      </tr>
                    ) : (
                      visibleMovements.map((movement) => (
                        <tr key={movement.id} className="border-t border-neutral-200 transition hover:bg-amber-50/50">
                          <td className="px-5 py-5 font-black text-neutral-950">
                            {movementTypeLabel(movement.type)}
                          </td>

                          <td className="px-5 py-5">
                            <div className="font-bold text-neutral-800">
                              {movement.concept || 'Movimiento'}
                            </div>
                            {movement.note && (
                              <div className="mt-1 text-xs text-neutral-400">
                                {movement.note}
                              </div>
                            )}
                          </td>

                          <td className="px-5 py-5 font-bold text-neutral-700">
                            {movement.type === 'PAYMENT_METHOD_TRANSFER'
                              ? `${methodLabel(movement.fromPaymentMethod)} → ${methodLabel(movement.toPaymentMethod)}`
                              : methodLabel(movement.paymentMethod)}
                          </td>

                          <td className={`px-5 py-5 font-black ${movementAmountClass(movement.type)}`}>
                            {formatMoney(movement.amount)}
                          </td>

                          <td className="px-5 py-5 text-xs font-bold text-neutral-500">
                            {formatDateTime(movement.movementDate)}
                          </td>

                          <td className="px-5 py-5">
                            <div className="flex flex-wrap justify-end gap-2">
                              <button
                                type="button"
                                onClick={() => setEditingMovement(movement)}
                                className="rounded-xl border border-neutral-200 bg-white px-3 py-2 text-xs font-black text-neutral-700 hover:bg-neutral-50"
                              >
                                Editar
                              </button>
                              <button
                                type="button"
                                onClick={() => handleDeleteMovement(movement)}
                                className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs font-black text-red-700 hover:bg-red-100"
                              >
                                Eliminar
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </section>

          <SalesSection
            sales={sales}
            canManageSales={canManageSales}
            labels={labels}
            onView={(sale) => setViewingSale(sale)}
            onEdit={(sale) => setEditingSale(sale)}
            onDelete={handleDeleteSale}
            fiscalRefreshKey={fiscalRefreshKey}
          />
        </>
      )}

      {showReconciliationModal && pendingReconciliation && selectedBranch && (
        <ReconciliationModal
          branch={selectedBranch}
          cashRegister={pendingReconciliation}
          canManageFund={canManageFund}
          onClose={() => setShowReconciliationModal(false)}
          onSaved={() => {
            setShowReconciliationModal(false);
            loadCash(selectedBranchId);
          }}
        />
      )}
      {showOpenModal && selectedBranch && (
        <OpenCashModal
          branch={selectedBranch}
          canManageFund={canManageFund}
          onClose={() => setShowOpenModal(false)}
          onSaved={() => {
            setShowOpenModal(false);
            loadCash(selectedBranchId);
          }}
        />
      )}

      {showCloseModal && selectedBranch && cashRegister && (
        <CloseCashModal
          branch={selectedBranch}
          cashRegister={cashRegister}
          canManageFund={canManageFund}
          onClose={() => setShowCloseModal(false)}
          onSaved={() => {
            setShowCloseModal(false);
            loadCash(selectedBranchId);
          }}
        />
      )}

      {showAuditActivityModal && selectedBranch && (
        <CashAuditActivityModal
          branch={selectedBranch}
          onClose={() => setShowAuditActivityModal(false)}
        />
      )}
      {showHistoryModal && selectedBranch && (
        <CashHistoryModal
          branch={selectedBranch}
          paymentMethods={paymentMethods}
          canManageFund={canManageFund}
          onClose={() => setShowHistoryModal(false)}
        />
      )}

      {showSaleModal && selectedBranch && cashRegister && (
        <SaleModal
          branch={selectedBranch}
          cashRegister={cashRegister}
          paymentMethods={paymentMethods}
          labels={labels}
          session={session}
          onClose={() => setShowSaleModal(false)}
          onSaved={async (createdSale, result = {}) => {
            setShowSaleModal(false);
            await loadCash(selectedBranchId);
            setFiscalRefreshKey((value) => value + 1);
            if (result.electronicDocumentError && createdSale) {
              const createdSaleId = saleIdOf(createdSale);
              if (createdSaleId && result.pendingElectronicDocument) {
                window.sessionStorage.setItem(
                  `gods.pending-electronic.${createdSaleId}`,
                  JSON.stringify(result.pendingElectronicDocument)
                );
              }
              setViewingSale({
                ...createdSale,
                electronicDocumentError: result.electronicDocumentError,
                pendingElectronicDocument: result.pendingElectronicDocument,
              });
            }
          }}
        />
      )}

      {showFundMovementModal && canManageFund && selectedBranch && (
        <FundMovementModal
          branch={selectedBranch}
          paymentMethods={paymentMethods}
          onClose={() => setShowFundMovementModal(false)}
          onSaved={async () => {
            setShowFundMovementModal(false);
            await loadCash(selectedBranchId);
          }}
        />
      )}

      {showMovementModal && selectedBranch && cashRegister && (
        <MovementModal
          branch={selectedBranch}
          cashRegister={cashRegister}
          paymentMethods={paymentMethods}
          canManageFund={canManageFund}
          onClose={() => setShowMovementModal(false)}
          onSaved={() => {
            setShowMovementModal(false);
            loadCash(selectedBranchId);
          }}
        />
      )}

      {editingMovement && selectedBranch && cashRegister && (
        <MovementModal
          branch={selectedBranch}
          cashRegister={cashRegister}
          paymentMethods={paymentMethods}
          canManageFund={canManageFund}
          initialMovement={editingMovement}
          onClose={() => setEditingMovement(null)}
          onSaved={() => {
            setEditingMovement(null);
            loadCash(selectedBranchId);
          }}
        />
      )}

      {showBarberPaymentModal && selectedBranch && cashRegister && (
        <BarberPaymentModal
          branch={selectedBranch}
          cashRegister={cashRegister}
          paymentMethods={paymentMethods}
          canManageFund={canManageFund}
          onClose={() => setShowBarberPaymentModal(false)}
          onSaved={() => {
            setShowBarberPaymentModal(false);
            loadCash(selectedBranchId);
          }}
        />
      )}

      {showAppointmentSaleModal && selectedBranch && cashRegister && pendingAppointment && (
        <AppointmentSaleModal
          branch={selectedBranch}
          cashRegister={cashRegister}
          appointment={pendingAppointment}
          paymentMethods={paymentMethods}
          labels={labels}
          session={session}
          onClose={() => setShowAppointmentSaleModal(false)}
          onSaved={() => {
            setShowAppointmentSaleModal(false);
            setPendingAppointment(null);
            loadCash(selectedBranchId);
          }}
        />
      )}


      {viewingSale && (
        <SaleDetailModal
          sale={viewingSale}
          branchId={selectedBranch?.id}
          onClose={() => setViewingSale(null)}
          onElectronicDocumentChanged={() => setFiscalRefreshKey((value) => value + 1)}
        />
      )}

      {editingSale && selectedBranch && (
        <EditSaleModal
          branch={selectedBranch}
          sale={editingSale}
          paymentMethods={paymentMethods}
          onClose={() => setEditingSale(null)}
          onSaved={() => {
            setEditingSale(null);
            loadCash(selectedBranchId);
          }}
        />
      )}
    </div>
  );
}

