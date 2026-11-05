from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether, Image
)
from pathlib import Path

OUT = Path(r"C:\ilustrativa_app\output\pdf\guia-comercial-super-gods-app.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)
LOGO = Path(r"C:\super-gods-landing\public\logo-super-gods.png")

NAVY = colors.HexColor("#0F2A5F")
BLUE = colors.HexColor("#2563EB")
GOLD = colors.HexColor("#D6A354")
INK = colors.HexColor("#0F172A")
MUTED = colors.HexColor("#526176")
PALE = colors.HexColor("#F3F6FB")
GREEN = colors.HexColor("#0F9D69")
WHITE = colors.white

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=29, leading=32, textColor=WHITE, alignment=TA_LEFT, spaceAfter=12))
styles.add(ParagraphStyle(name="CoverSub", parent=styles["BodyText"], fontName="Helvetica", fontSize=12.5, leading=18, textColor=colors.HexColor("#DCE7FA")))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=NAVY, spaceAfter=8))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=INK, spaceBefore=3, spaceAfter=4))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.4, leading=13.5, textColor=MUTED, spaceAfter=5))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=8, leading=11, textColor=MUTED))
styles.add(ParagraphStyle(name="CardTitle", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=11.2, leading=14, textColor=NAVY, spaceAfter=4))
styles.add(ParagraphStyle(name="CardBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=MUTED))
styles.add(ParagraphStyle(name="Quote", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=13, leading=19, textColor=NAVY, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="Call", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=15, textColor=INK))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2EF"))
    canvas.line(18*mm, 13*mm, 192*mm, 13*mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18*mm, 8.5*mm, "GODS TECHNOLOGIES S.A.C. | Super Gods App | WhatsApp +51 958 062 847")
    canvas.drawRightString(192*mm, 8.5*mm, f"Pagina {doc.page}")
    canvas.restoreState()

def card(title, body, width=82*mm):
    data = [[Paragraph(title, styles["CardTitle"])], [Paragraph(body, styles["CardBody"])]]
    t = Table(data, colWidths=[width], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), WHITE),
        ("BOX", (0,0), (-1,-1), 0.7, colors.HexColor("#DCE4EF")),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,0), 9),
        ("BOTTOMPADDING", (0,0), (-1,0), 1),
        ("TOPPADDING", (0,1), (-1,1), 2),
        ("BOTTOMPADDING", (0,1), (-1,1), 9),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    return t

def section_label(text):
    t = Table([[Paragraph(text.upper(), ParagraphStyle(name="label", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=WHITE))]], colWidths=[45*mm])
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), BLUE), ("LEFTPADDING",(0,0),(-1,-1),8), ("RIGHTPADDING",(0,0),(-1,-1),8), ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    return t

story = []

# Cover
cover = Table([["", ""]], colWidths=[104*mm, 70*mm], rowHeights=[235*mm])
cover.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), NAVY), ("BOX", (0,0), (-1,-1), 0, NAVY)]))
cover_left = []
if LOGO.exists():
    cover_left.append(Image(str(LOGO), width=25*mm, height=25*mm))
    cover_left.append(Spacer(1, 16))
cover_left += [
    Paragraph("GUIA COMERCIAL PARA LLAMADAS", ParagraphStyle(name="eyebrow", fontName="Helvetica-Bold", fontSize=8.5, leading=10, textColor=GOLD, spaceAfter=10)),
    Paragraph("Super Gods App", styles["CoverTitle"]),
    Paragraph("Un resumen practico para explicar la plataforma a duenos de barberias, salones, spas y centros de belleza.", styles["CoverSub"]),
    Spacer(1, 18),
    Paragraph("Mas reservas. Caja clara. Clientes que vuelven.", ParagraphStyle(name="promise", fontName="Helvetica-Bold", fontSize=15, leading=20, textColor=WHITE)),
    Spacer(1, 18),
    Paragraph("Prueba gratuita por 7 dias", ParagraphStyle(name="trial", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#BEEAD6"))),
]
cover_right = [
    Spacer(1, 35),
    Paragraph("La idea central", ParagraphStyle(name="mini", fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=GOLD)),
    Spacer(1, 8),
    Paragraph("Super Gods App centraliza agenda, ventas, caja, clientes, inventario, equipo y reportes para que el negocio trabaje con orden y pueda crecer.", styles["CoverSub"]),
    Spacer(1, 22),
    Paragraph("Ideal para", ParagraphStyle(name="mini2", fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=GOLD)),
    Spacer(1, 8),
    Paragraph("- Barberias<br/>- Salones de belleza<br/>- Peluquerias<br/>- Spas y estetica<br/>- Unas, cejas y pestanas<br/>- Negocios con una o varias sedes", styles["CoverSub"]),
]
cover._cellvalues[0][0] = cover_left
cover._cellvalues[0][1] = cover_right
cover.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"), ("LEFTPADDING",(0,0),(-1,-1),18), ("RIGHTPADDING",(0,0),(-1,-1),18), ("TOPPADDING",(0,0),(-1,-1),22)]))
story += [cover, PageBreak()]

# Page 2
story += [section_label("Que resuelve"), Spacer(1, 8), Paragraph("Todo el negocio en una sola plataforma", styles["H1x"]), Paragraph("No necesitas explicar cada boton. Enfoca la llamada en los problemas que el cliente ya vive y conecta cada problema con una solucion concreta.", styles["Bodyx"]), Spacer(1, 8)]
rows = [
    [card("Agenda y reservas", "Reservas online las 24 horas, agenda por profesional y sede, horarios disponibles, bloqueos, reprogramaciones, cancelaciones, anticipos y recordatorios."), card("Caja y ventas", "Ventas de servicios y productos, efectivo, Yape, Plin, tarjeta y pagos mixtos; ingresos, gastos, apertura, cierre, historial y auditoria.")],
    [card("Clientes y fidelizacion", "Ficha del cliente, historial, visitas, puntos, premios, categorias personalizadas, promociones y seguimiento para aumentar la frecuencia de regreso."), card("Inventario y productos", "Catalogo de productos, stock, movimientos, alertas, ventas, pedidos y control para reducir faltantes o perdidas.")],
    [card("Equipo y comisiones", "Profesionales por sede, servicios habilitados, horarios, bloqueos, comision general o por servicio, adelantos, pagos y liquidaciones."), card("Reportes para decidir", "Ventas, gastos, utilidad, metodos de pago, ticket promedio, rendimiento por sede y profesional, clientes y exportacion a Excel o PDF.")],
]
t = Table(rows, colWidths=[86*mm,86*mm], hAlign="LEFT")
t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"), ("LEFTPADDING",(0,0),(-1,-1),0), ("RIGHTPADDING",(0,0),(-1,-1),5), ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
story += [t, Spacer(1, 8), Table([[Paragraph("Frase recomendada", styles["CardTitle"]), Paragraph("La aplicacion te ayuda a ordenar las citas, saber cuanto vendiste realmente, controlar a tu equipo y hacer que tus clientes regresen.", styles["Quote"])]], colWidths=[40*mm,132*mm], style=[("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#EAF2FF")),("BOX",(0,0),(-1,-1),0.8,colors.HexColor("#BBD0F5")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10)]), PageBreak()]

# Page 3
story += [section_label("Mas funciones"), Spacer(1,8), Paragraph("Control, seguridad y crecimiento", styles["H1x"])]
more = [
    ("App movil y panel web", "El dueno controla desde computadora o celular. Profesionales, administradores y cajeros ven solo lo que corresponde a su rol."),
    ("Multi-sede", "Cada sede puede tener agenda, caja, profesionales, horarios, comisiones y reportes propios; el dueno ve tambien el consolidado."),
    ("Roles y permisos", "Accesos para dueno, administrador, cajero y profesional. Se limita informacion sensible y se controla la sede operativa."),
    ("Auditoria", "Registro de cambios importantes: ventas, gastos, horarios, permisos, profesionales y otros movimientos sensibles."),
    ("Promociones y notificaciones", "Campanas con imagen, notificaciones push, avisos dentro de la app, premios y promociones para reactivar clientes."),
    ("Reserva publica y enlaces", "El negocio comparte su enlace o codigo para que el cliente reserve sin depender de llamadas o mensajes desordenados."),
    ("Comprobantes electronicos - SUNAT", "Modulo para emitir boletas y facturas electronicas mediante proveedor autorizado, consultar estado y abrir o imprimir el comprobante oficial. Se configura aparte y requiere credenciales de produccion."),
    ("Software a medida", "GODS TECHNOLOGIES S.A.C. tambien desarrolla sistemas web, aplicaciones moviles, automatizaciones e integraciones para empresas."),
]
for i in range(0, len(more), 2):
    story.append(Table([[card(more[i][0], more[i][1]), card(more[i+1][0], more[i+1][1])]], colWidths=[86*mm,86*mm], style=[("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
story += [Spacer(1,8), Paragraph("Nota comercial", styles["H2x"]), Paragraph("Las funciones, limites de sedes, cantidad de profesionales, administradores y modulos premium dependen del plan contratado. Durante la demo se recomienda configurar el negocio segun su operacion real.", styles["Bodyx"]), PageBreak()]

# Page 4
story += [section_label("Guion telefonico"), Spacer(1,8), Paragraph("Una llamada simple de 5 pasos", styles["H1x"])]
steps = [
    ("1. Presentacion", "Hola, soy [tu nombre] de Super Gods App. Ayudamos a barberias y salones a ordenar citas, caja, clientes e inventario desde el celular y la computadora."),
    ("2. Diagnostico", "Antes de explicarte todo, quisiera saber: ¿como manejas hoy tus reservas y tu caja?"),
    ("3. Conectar el problema", "Por lo que me comentas, Super Gods App te ayudaria especialmente con [agenda / caja / comisiones / inventario / clientes]."),
    ("4. Demostracion breve", "El cliente reserva, el equipo atiende y registra la venta, y tu puedes revisar caja, comisiones, inventario y reportes en tiempo real."),
    ("5. Cierre", "Podemos activarte una prueba gratuita por 7 dias y agendar una explicacion para dejarla configurada. ¿Te parece si empezamos con tus datos?"),
]
for title, body in steps:
    story.append(KeepTogether([Paragraph(title, styles["H2x"]), Table([[Paragraph(body, styles["Call"])]], colWidths=[172*mm], style=[("BACKGROUND",(0,0),(-1,-1),PALE),("BOX",(0,0),(-1,-1),0.6,colors.HexColor("#DCE4EF")),("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8)]), Spacer(1,6)]))
story += [Spacer(1,4), Paragraph("Preguntas para descubrir la necesidad", styles["H2x"]), Paragraph("- ¿Cuantas citas manejas al dia?<br/>- ¿Trabajas con una o varias sedes?<br/>- ¿Cuantos profesionales tienes?<br/>- ¿Como calculas comisiones y pagos?<br/>- ¿Controlas productos y stock?<br/>- ¿Tus clientes reservan por WhatsApp?<br/>- ¿Emites boleta o factura electronica?<br/>- ¿Que es lo que mas tiempo te hace perder hoy?", styles["Bodyx"]), PageBreak()]

# Page 5
story += [section_label("Objeciones y cierre"), Spacer(1,8), Paragraph("Respuestas rapidas para la llamada", styles["H1x"])]
objections = [
    ("Ya uso WhatsApp", "WhatsApp ayuda a conversar, pero Super Gods organiza horarios, evita cruces y deja la informacion conectada con ventas, caja y clientes."),
    ("Mi negocio es pequeno", "Precisamente por eso conviene empezar ordenado. Puedes usar un plan inicial y crecer sin cambiar de sistema."),
    ("Mi equipo no sabe usar sistemas", "La operacion esta pensada para celular y cada persona ve solo sus funciones. La activacion incluye acompanamiento."),
    ("No quiero perder mis datos", "Se configura el catalogo y los clientes con cuidado. Ademas, existen exportaciones e historial para mantener control."),
    ("Quiero pensarlo", "Claro. Lo mejor es probarlo con tu negocio durante 7 dias; asi decides con datos reales y sin tarjeta."),
    ("¿Incluye SUNAT?", "Existe emision de boleta y factura electronica. Se habilita como modulo aparte y requiere configurar los accesos del proveedor autorizado."),
]
for q,a in objections:
    story.append(Table([[Paragraph(q, styles["CardTitle"]), Paragraph(a, styles["CardBody"])]], colWidths=[48*mm,124*mm], style=[("BOX",(0,0),(-1,-1),0.6,colors.HexColor("#DCE4EF")),("BACKGROUND",(0,0),(0,0),colors.HexColor("#EAF2FF")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),9),("RIGHTPADDING",(0,0),(-1,-1),9),("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8)]))
    story.append(Spacer(1,5))
story += [Spacer(1,8), Paragraph("Datos para activar la prueba", styles["H2x"]), Paragraph("- Nombre comercial del negocio<br/>- Tipo de negocio<br/>- Nombre completo del dueno<br/>- Correo de acceso<br/>- Numero de WhatsApp<br/>- Pais y ciudad<br/>- Cantidad de sedes<br/>- Cantidad de profesionales<br/>- Horario disponible para una reunion por Google Meet", styles["Bodyx"]), Spacer(1,8)]
cta = Table([[Paragraph("PROXIMO PASO", ParagraphStyle(name="cta1", fontName="Helvetica-Bold", fontSize=9, textColor=GOLD)), Paragraph("Activar 7 dias gratis y agendar la configuracion", ParagraphStyle(name="cta2", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=WHITE))]], colWidths=[38*mm,134*mm])
cta.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),NAVY),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),12)]))
story.append(cta)

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=17*mm, bottomMargin=18*mm, title="Guia comercial Super Gods App", author="GODS TECHNOLOGIES S.A.C.")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
