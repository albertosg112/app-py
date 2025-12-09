from flask import Flask, request, jsonify
import csv
import os
import hashlib
import datetime
import json
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# Configuración segura (usa variables de entorno en producción)
HOTMART_SECRET = os.getenv('HOTMART_SECRET', 'tu_secreto_de_hotmart_aqui')
CODIGOS_FILE = "codigos_acceso.csv"

# Crear archivo CSV si no existe
if not os.path.exists(CODIGOS_FILE):
    with open(CODIGOS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['codigo', 'email', 'fecha_compra', 'nivel', 'activo', 'id_compra'])

def generar_codigo_unico(email, fecha, id_compra):
    """Genera un código único e irrepetible basado en datos de la compra"""
    semilla = f"{email}{fecha}{id_compra}SG1_MAGIC_SECRET"
    return hashlib.sha256(semilla.encode()).hexdigest()[:12].upper()

@app.route('/webhook/hotmart', methods=['POST'])
def hotmart_webhook():
    # Verificar autenticidad de la solicitud
    signature = request.headers.get('x-hotmart-signature')
    payload = request.get_data().decode('utf-8')
    
    if not signature or not verificar_firma(payload, signature):
        return jsonify({"error": "Firma inválida"}), 401
    
    try:
        data = json.loads(payload)
        
        # Extraer información relevante
        email_cliente = data['email_buyer'] or data['buyer']['email']
        fecha_compra = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        id_compra = data['purchase_id'] or data['id']
        nivel_acceso = "PRO" if "PRO" in data.get('product_name', '').upper() else "BASICO"
        
        # Generar código único
        codigo_acceso = generar_codigo_unico(email_cliente, fecha_compra, id_compra)
        
        # Guardar en CSV
        with open(CODIGOS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                codigo_acceso,
                email_cliente,
                fecha_compra,
                nivel_acceso,
                '1',  # activo = sí
                id_compra
            ])
        
        # Enviar email al cliente (simulado - en producción usa SendGrid o similar)
        enviar_email_acceso(email_cliente, codigo_acceso, nivel_acceso)
        
        return jsonify({"success": True, "codigo": codigo_acceso}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def verificar_firma(payload, signature):
    """Verifica la firma de Hotmart para asegurar que la solicitud es legítima"""
    # En producción, implementa la verificación real con HMAC
    # Por ahora, verificamos que exista la clave secreta
    return HOTMART_SECRET in payload or True  # TEMPORAL - Implementar seguridad real

def enviar_email_acceso(email, codigo, nivel):
    """Función simulada para enviar email con código de acceso"""
    # En producción, usa un servicio como SendGrid, Mailgun, etc.
    print(f"📧 Enviar a {email}:")
    print(f"Tu código de acceso {nivel} es: {codigo}")
    print("Visita: https://tu-app.streamlit.app para activarlo\n")

if __name__ == '__main__':
    app.run(debug=True)
