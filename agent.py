import os
import asyncio
import logging
import httpx
from datetime import datetime, timedelta
import pytz
import random
import re

from dotenv import load_dotenv
from typing_extensions import Annotated

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    RoomInputOptions,
    function_tool,
    RunContext,
    llm
)

from livekit.plugins import (
    cartesia,
    openai,
    deepgram,
    silero,
    elevenlabs
)

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

# --- Variables Globales ---
current_participant_info = {"participant_id": None, "phone_number": None, "session_id": None}
active_session: AgentSession | None = None

# 🔧 Estado de consulta mejorado para soportar agendamiento
consultation_state = {
    "is_active": False,
    "current_query": None,
    "operation_type": None,
    "start_time": None,
    "task": None,
    "feedback_task": None,
    "feedback_sent": [],
    "last_completed": None
}

def get_greeting():
    """Retorna un saludo apropiado según la hora en Perú."""
    hour = datetime.now(pytz.timezone("America/Lima")).hour
    if 5 <= hour < 12: return "Buenos días"
    elif 12 <= hour < 19: return "Buenas tardes"
    else: return "Buenas noches"

def get_farewell():
    """Retorna la despedida apropiada según la hora"""
    hour = datetime.now(pytz.timezone("America/Lima")).hour
    if 5 <= hour < 12: return "Que tenga un buen día"
    elif 12 <= hour < 19: return "Que tenga una buena tarde"
    else: return "Que tenga una buena noche"

def extract_phone_from_identity(identity: str) -> str | None:
    """Extrae un número de teléfono del identificador de participante de SIP."""
    if not identity: return None
    try:
        identity = identity.strip()
        if identity.startswith('sip_'):
            phone = identity.replace('sip_', '')
            return phone if phone.startswith('+') else '+51' + phone
        if 'sip:' in identity:
            parts = identity.split('sip:')[1].split('@')[0]
            return parts if parts.startswith('+') else '+51' + parts
        return identity
    except Exception as e:
        logger.warning(f"No se pudo extraer el teléfono de '{identity}': {e}")
        return identity

def normalize_phone(phone: str | None) -> tuple[str | None, str | None]:
    """Normaliza el número de teléfono y valida su formato."""
    if not phone:
        return None, "Por favor, proporciona un número de celular válido."
    
    # Limpiar espacios, guiones y otros caracteres
    phone = re.sub(r'[\s\-\(\)]+', '', phone.strip())
    
    # Si el número no comienza con '+', asumir código de país +51 (Perú)
    if not phone.startswith('+'):
        phone = '+51' + phone
    
    # Validar formato: +51 seguido de 9 dígitos
    if re.match(r'^\+51\d{9}$', phone):
        return phone, None
    
    return None, "El número de celular no es válido. Debe tener 9 dígitos, por ejemplo, +51987654321."

def parse_relative_date(date_input: str | None, day_reference: str | None = None) -> tuple[str | None, str | None]:
    """
    Convierte fechas relativas como 'mañana' a formato DD/MM/YYYY, validando el día de la semana.
    Retorna (fecha_formateada, mensaje_error).
    """
    logger.debug(f"[DATE] Procesando fecha: date_input='{date_input}', day_reference='{day_reference}'")
    
    if not date_input:
        logger.warning("[DATE] No se proporcionó fecha")
        return None, "Por favor, proporciona la fecha de la cita."
    
    date_input = date_input.lower().strip()
    lima_tz = pytz.timezone("America/Lima")
    today = datetime.now(lima_tz)
    
    # Mapa de días en español a índices (0=lunes, 6=domingo)
    days_map = {
        "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
        "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6
    }
    
    if "mañana" in date_input or "manana" in date_input:
        tomorrow = today + timedelta(days=1)
        expected_day = tomorrow.weekday()
        
        logger.debug(f"[DATE] Detectado 'mañana', fecha calculada: {tomorrow.strftime('%d/%m/%Y')}, día: {tomorrow.strftime('%A').lower()}")
        
        if day_reference:
            day_reference = day_reference.lower().strip()
            if day_reference in days_map:
                if days_map[day_reference] != expected_day:
                    error_msg = f"Mañana es {tomorrow.strftime('%A').lower()}, no {day_reference}. Por favor, especifica una fecha válida."
                    logger.warning(f"[DATE] Error de validación: {error_msg}")
                    return None, error_msg
        
        return tomorrow.strftime("%d/%m/%Y"), None
    
    # Validar formato DD/MM/YYYY
    try:
        parsed_date = datetime.strptime(date_input, "%d/%m/%Y").replace(tzinfo=lima_tz)
        if parsed_date.date() < today.date():
            logger.warning(f"[DATE] Fecha pasada detectada: {date_input}")
            return None, "La fecha no puede ser anterior a hoy. Por favor, especifica una fecha válida."
        
        if day_reference:
            day_reference = day_reference.lower().strip()
            if day_reference in days_map:
                if days_map[day_reference] != parsed_date.weekday():
                    error_msg = f"La fecha {date_input} es un {parsed_date.strftime('%A').lower()}, no {day_reference}. Por favor, especifica una fecha válida."
                    logger.warning(f"[DATE] Error de validación: {error_msg}")
                    return None, error_msg
        
        logger.debug(f"[DATE] Fecha válida: {date_input}")
        return parsed_date.strftime("%d/%m/%Y"), None
    except ValueError:
        logger.warning(f"[DATE] Formato de fecha inválido: {date_input}")
        return None, "No entendí la fecha. Por favor, usa el formato día/mes/año, como 05/08/2025, o di 'mañana'."

elevenlabs_tts = elevenlabs.TTS(
    voice_id="VywzfvxxNk4yFAaoMm4Q",
    model="eleven_turbo_v2_5",
    api_key=os.getenv("ELEVEN_API_KEY")
)

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Eres 'Mia', una asistente de voz de Multiservicioscall con acento peruano. "
                "La conversación debe ser respetuosa y cortés. "
                "Tu interfaz con los usuarios será por voz. "
                "Usa respuestas cortas y claras, evita signos de puntuación impronunciables y al listar evita mencionar el número de item. "
                "SIEMPRE responde a cada mensaje del usuario, nunca los ignores. "
                "Si el usuario menciona cualquier servicio (telemarketing, cobranza, encuestas, digitación, etc.), usa INMEDIATAMENTE la herramienta 'query_knowledge_base'. "
                "Si el usuario solicita agendar una cita con el gerente comercial, usa INMEDIATAMENTE la herramienta 'schedule_appointment' y recolecta nombre completo, número de celular (9 dígitos, por ejemplo, 987654321), fecha y hora. "
                "Si el usuario dice 'mañana' u otra fecha relativa, pásala directamente a 'schedule_appointment' sin intentar convertirla. La herramienta interpretará 'mañana' según la fecha actual en Lima, Perú. "
                "Si el usuario proporciona una fecha explícita, debe estar en formato DD/MM/YYYY. "
                "Valida que el número de celular sea correcto (9 dígitos, opcionalmente con +51). "
                "Si faltan datos o son incorrectos, pide clarificación específica. "
                "Confirma los datos recibidos antes de agendar (por ejemplo, 'Entiendo, quieres una cita para [nombre] el [fecha] a las [hora]. ¿Es correcto?'). "
                "Si el usuario dice solo 'Hola' o pregunta si estás ahí, responde inmediatamente que sí estás presente. "
                "Si no entiendes algo, pide clarificación inmediatamente. "
                "NUNCA dejes al usuario hablando en el vacío - siempre confirma que has escuchado. "
                "Para CUALQUIER consulta sobre servicios o la empresa, usa la herramienta 'query_knowledge_base'. "
                "Si el usuario expresa impaciencia, usa la herramienta 'handle_user_impatience'."
            ),
            stt=deepgram.STT(model="nova", language="es-419"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(
                model="sonic-2",
                language="es",
                voice="5c5ad5e7-1020-476b-8b91-fdcbe9cc313c"
            ),
        )
        logger.info(f"[TOOLS] Registradas: {[t.__name__ for t in self._tools]}")

    @function_tool()
    async def query_knowledge_base(
        self, 
        ctx: RunContext,
        question: Annotated[str, "Pregunta del usuario sobre productos, servicios o información de la empresa."]
    ) -> str:
        """
        🔧 VERSIÓN CORREGIDA: Mejor manejo de concurrencia y feedback sin conflictos
        """
        global consultation_state
        
        if consultation_state["is_active"]:
            logger.info(f"[TOOL] Nueva pregunta durante consulta activa: {question}")
            
            if consultation_state["feedback_task"] and not consultation_state["feedback_task"].done():
                consultation_state["feedback_task"].cancel()
                logger.info("[TOOL] Feedback anterior cancelado")
            
            consultation_state["current_query"] = question
            consultation_state["operation_type"] = "rag"
            
            if consultation_state["task"]:
                try:
                    result = await consultation_state["task"]
                    return result
                except asyncio.CancelledError:
                    logger.info("[TOOL] Consulta anterior cancelada")

        consultation_state.update({
            "is_active": True,
            "current_query": question,
            "operation_type": "rag",
            "start_time": asyncio.get_event_loop().time(),
            "task": None,
            "feedback_task": None,
            "feedback_sent": []
        })
        
        logger.info(f"[TOOL] Iniciando consulta para: {question}")

        async def controlled_feedback():
            """
            🔧 FEEDBACK MEJORADO: Usa variaciones de locución para mensajes de espera
            y evita conflictos con el sistema de silencios
            """
            try:
                consultation_start = consultation_state["start_time"]
                
                initial_messages = [
                    "Consultando la información, un momento por favor...",
                    "Buscando los detalles para ti, espera un segundo...",
                    "Procesando tu consulta, dame un momento..."
                ]
                await asyncio.sleep(3.0)
                if consultation_state["is_active"] and active_session:
                    message = random.choice(initial_messages)
                    await active_session.say(message)
                    consultation_state["feedback_sent"].append("initial")
                    logger.info(f"[TOOL] Feedback 1: Mensaje inicial enviado: '{message}'")
                
                processing_messages = [
                    "Un momento más, estoy procesando tu consulta...",
                    "Ya casi tengo la respuesta, espera un poco por favor...",
                    "Seguimos buscando la información, gracias por esperar..."
                ]
                await asyncio.sleep(6.0)
                if consultation_state["is_active"] and active_session:
                    message = random.choice(processing_messages)
                    await active_session.say(message)
                    consultation_state["feedback_sent"].append("processing")
                    logger.info(f"[TOOL] Feedback 2: Mensaje de procesamiento enviado: '{message}'")
                
                patience_messages = [
                    "Aún estoy buscando la información, gracias por tu paciencia...",
                    "La respuesta está en camino, te agradezco la espera...",
                    "Seguimos procesando, un poco más y te respondo..."
                ]
                while consultation_state["is_active"]:
                    await asyncio.sleep(8.0)
                    if consultation_state["is_active"] and active_session:
                        current_time = asyncio.get_event_loop().time()
                        elapsed = current_time - consultation_start
                        message = random.choice(patience_messages)
                        await active_session.say(message)
                        consultation_state["feedback_sent"].append("patience")
                        logger.info(f"[TOOL] Feedback paciencia después de {elapsed:.1f}s: '{message}'")
                        
            except asyncio.CancelledError:
                logger.info("[TOOL] Feedback controlado cancelado")
            except Exception as e:
                logger.error(f"[TOOL] Error en feedback controlado: {e}")

        consultation_state["feedback_task"] = asyncio.create_task(controlled_feedback())

        try:
            async def perform_query():
                """Realiza la consulta real al RAG"""
                N8N_RAG_WEBHOOK_URL = os.getenv("N8N_RAG_WEBHOOK_URL")
                if not N8N_RAG_WEBHOOK_URL:
                    logger.error("[TOOL] La URL del webhook de N8N no está configurada.")
                    return "Lo siento, hay un problema de configuración que me impide buscar la información."

                current_question = consultation_state["current_query"]
                payload = {"question": current_question.strip(), "caller_info": current_participant_info}
                logger.info(f"[TOOL] Enviando payload a N8N: {payload}")

                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.post(N8N_RAG_WEBHOOK_URL, json=payload)

                response.raise_for_status()
                data = response.json()
                logger.info(f"[TOOL] Datos recibidos de N8N: {data}")

                answer = data.get("answer", "No encontré una respuesta clara a tu consulta.")
                return answer.replace("**", "").replace("*", "").replace("📞", "").strip()

            consultation_state["task"] = asyncio.create_task(perform_query())
            result = await consultation_state["task"]
            
            consultation_time = asyncio.get_event_loop().time() - consultation_state["start_time"]
            logger.info(f"[TOOL] Consulta completada en {consultation_time:.2f} segundos")
            
            return result

        except asyncio.CancelledError:
            logger.info("[TOOL] Consulta cancelada")
            return "He recibido tu nueva pregunta, déjame procesarla."
            
        except httpx.TimeoutException:
            logger.error("[TOOL] Timeout al consultar N8N.")
            return "La consulta está tardando más de lo esperado. ¿Podrías repetir tu pregunta o ser más específico?"
            
        except httpx.HTTPStatusError as e:
            logger.error(f"[TOOL] Error HTTP de N8N: {e.response.status_code}")
            return "No pude conectarme con el sistema de información en este momento. ¿Puedo ayudarte con algo más?"
            
        except Exception as e:
            logger.error(f"[TOOL] Error inesperado: {e}")
            return "Ocurrió un error al procesar tu solicitud. ¿Podrías intentar reformular tu pregunta?"
        
        finally:
            logger.info("[TOOL] Limpiando estado de consulta...")
            
            if consultation_state["feedback_task"] and not consultation_state["feedback_task"].done():
                consultation_state["feedback_task"].cancel()
                try:
                    await consultation_state["feedback_task"]
                except asyncio.CancelledError:
                    pass
                logger.info("[TOOL] Feedback task cancelado")
            
            consultation_state.update({
                "is_active": False,
                "current_query": None,
                "operation_type": None,
                "start_time": None,
                "task": None,
                "feedback_task": None,
                "last_completed": asyncio.get_event_loop().time(),
                "feedback_sent": []
            })
            
            logger.info("[TOOL] Estado de consulta limpio")

    @function_tool()
    async def schedule_appointment(
        self,
        ctx: RunContext,
        name: Annotated[str | None, "Nombre completo del usuario"] = None,
        phone: Annotated[str | None, "Número de celular del usuario"] = None,
        date: Annotated[str | None, "Fecha de la cita (puede ser 'mañana' o formato DD/MM/YYYY)"] = None,
        day_reference: Annotated[str | None, "Día de la semana opcional (ej. 'lunes') para validar fechas relativas"] = None,
        time: Annotated[str | None, "Hora de la cita en formato HH:MM (hora de Lima, Perú)"] = None,
        reason: Annotated[str | None, "Motivo de la cita (opcional)"] = None
    ) -> str:
        """
        🔧 Herramienta para agendar citas con el gerente comercial vía Google Calendar usando n8n
        """
        global consultation_state
        
        logger.debug(f"[SCHEDULE] Recibido: name='{name}', phone='{phone}', date='{date}', day_reference='{day_reference}', time='{time}', reason='{reason}'")
        
        # Normalizar número de teléfono
        normalized_phone, phone_error = normalize_phone(phone)
        if phone_error:
            logger.warning(f"[SCHEDULE] Error en número de teléfono: {phone_error}")
            return phone_error
        
        # Procesar fecha relativa
        formatted_date, date_error = parse_relative_date(date, day_reference)
        if date_error:
            logger.warning(f"[SCHEDULE] Error en fecha: {date_error}")
            return date_error
        
        # Verificar datos requeridos
        missing_data = []
        if not name:
            missing_data.append("nombre completo")
        if not normalized_phone:
            missing_data.append("número de celular")
        if not formatted_date:
            missing_data.append("fecha")
        if not time:
            missing_data.append("hora")

        if missing_data:
            error_msg = f"Por favor, proporciona los siguientes datos para agendar la cita: {', '.join(missing_data)}."
            logger.warning(f"[SCHEDULE] Datos faltantes: {error_msg}")
            return error_msg

        # Confirmar datos al usuario
        confirmation_msg = f"Entiendo, quieres una cita para {name} el {formatted_date} a las {time}. ¿Es correcto?"
        await active_session.say(confirmation_msg)
        logger.info(f"[SCHEDULE] Confirmación enviada: {confirmation_msg}")

        if consultation_state["is_active"]:
            logger.info(f"[SCHEDULE] Nueva solicitud de cita durante operación activa: {formatted_date} {time}")
            
            if consultation_state["feedback_task"] and not consultation_state["feedback_task"].done():
                consultation_state["feedback_task"].cancel()
                logger.info("[SCHEDULE] Feedback anterior cancelado")
            
            consultation_state["current_query"] = f"Agendar cita para {name} el {formatted_date} a las {time}"
            consultation_state["operation_type"] = "appointment"
            
            if consultation_state["task"]:
                try:
                    result = await consultation_state["task"]
                    return result
                except asyncio.CancelledError:
                    logger.info("[SCHEDULE] Operación anterior cancelada")

        consultation_state.update({
            "is_active": True,
            "current_query": f"Agendar cita para {name} el {formatted_date} a las {time}",
            "operation_type": "appointment",
            "start_time": asyncio.get_event_loop().time(),
            "task": None,
            "feedback_task": None,
            "feedback_sent": []
        })
        
        logger.info(f"[SCHEDULE] Iniciando agendamiento para: {name}, {formatted_date} {time}, motivo: {reason}")

        async def controlled_feedback():
            """
            🔧 Feedback con variaciones de locución durante el agendamiento
            """
            try:
                consultation_start = consultation_state["start_time"]
                
                initial_messages = [
                    "Procesando tu solicitud de cita, un momento por favor...",
                    "Agendando tu cita con el gerente comercial, espera un segundo...",
                    "Verificando la disponibilidad, dame un momento..."
                ]
                await asyncio.sleep(3.0)
                if consultation_state["is_active"] and active_session:
                    message = random.choice(initial_messages)
                    await active_session.say(message)
                    consultation_state["feedback_sent"].append("initial")
                    logger.info(f"[SCHEDULE] Feedback 1: Mensaje inicial enviado: '{message}'")
                
                processing_messages = [
                    "Un momento más, estoy confirmando la cita...",
                    "Ya casi terminamos de agendar, gracias por esperar...",
                    "Procesando los detalles de tu cita, un segundo más..."
                ]
                await asyncio.sleep(6.0)
                if consultation_state["is_active"] and active_session:
                    message = random.choice(processing_messages)
                    await active_session.say(message)
                    consultation_state["feedback_sent"].append("processing")
                    logger.info(f"[SCHEDULE] Feedback 2: Mensaje de procesamiento enviado: '{message}'")
                
                patience_messages = [
                    "Aún estoy coordinando la cita, gracias por tu paciencia...",
                    "La cita está en proceso, te agradezco la espera...",
                    "Seguimos trabajando en tu solicitud, un poco más y listo..."
                ]
                while consultation_state["is_active"]:
                    await asyncio.sleep(8.0)
                    if consultation_state["is_active"] and active_session:
                        current_time = asyncio.get_event_loop().time()
                        elapsed = current_time - consultation_start
                        message = random.choice(patience_messages)
                        await active_session.say(message)
                        consultation_state["feedback_sent"].append("patience")
                        logger.info(f"[SCHEDULE] Feedback paciencia después de {elapsed:.1f}s: '{message}'")
                        
            except asyncio.CancelledError:
                logger.info("[SCHEDULE] Feedback controlado cancelado")
            except Exception as e:
                logger.error(f"[SCHEDULE] Error en feedback controlado: {e}")

        consultation_state["feedback_task"] = asyncio.create_task(controlled_feedback())

        try:
            async def perform_scheduling():
                N8N_APPOINTMENT_WEBHOOK_URL = os.getenv("N8N_APPOINTMENT_WEBHOOK_URL")
                if not N8N_APPOINTMENT_WEBHOOK_URL:
                    logger.error("[SCHEDULE] La URL del webhook de citas de N8N no está configurada.")
                    return "Lo siento, hay un problema de configuración que me impide agendar la cita."

                payload = {
                    "caller_info": current_participant_info,
                    "name": name.strip(),
                    "phone": normalized_phone.strip(),
                    "date": formatted_date.strip(),
                    "time": time.strip(),
                    "reason": reason.strip() if reason else "Cita con gerente comercial"
                }
                logger.info(f"[SCHEDULE] Enviando payload de cita a N8N: {payload}")

                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.post(N8N_APPOINTMENT_WEBHOOK_URL, json=payload)

                response.raise_for_status()
                data = response.json()
                logger.info(f"[SCHEDULE] Datos recibidos de N8N: {data}")

                if data.get("status") == "success":
                    return f"¡Cita agendada exitosamente para {name} el {formatted_date} a las {time}! Recibirás una confirmación pronto. ¿En qué más puedo ayudarte?"
                else:
                    return f"Lo siento, no se pudo agendar la cita para {name}. ¿Podrías intentar con otra fecha u hora?"

            consultation_state["task"] = asyncio.create_task(perform_scheduling())
            result = await consultation_state["task"]
            
            consultation_time = asyncio.get_event_loop().time() - consultation_state["start_time"]
            logger.info(f"[SCHEDULE] Agendamiento completado en {consultation_time:.2f} segundos")
            
            return result

        except asyncio.CancelledError:
            logger.info("[SCHEDULE] Agendamiento cancelado")
            return "He recibido una nueva solicitud, déjame procesarla."
            
        except httpx.TimeoutException:
            logger.error("[SCHEDULE] Timeout al agendar con N8N.")
            return "El proceso de agendamiento está tomando más tiempo del esperado. ¿Podrías intentar de nuevo?"
            
        except httpx.HTTPStatusError as e:
            logger.error(f"[SCHEDULE] Error HTTP de N8N: {e.response.status_code}")
            return "No pude conectar con el sistema de agendamiento. ¿Puedo ayudarte con algo más?"
            
        except Exception as e:
            logger.error(f"[SCHEDULE] Error inesperado: {e}")
            return "Ocurrió un error al agendar tu cita. ¿Podrías intentar de nuevo o especificar otra fecha?"
        
        finally:
            logger.info("[SCHEDULE] Limpiando estado de agendamiento...")
            
            if consultation_state["feedback_task"] and not consultation_state["feedback_task"].done():
                consultation_state["feedback_task"].cancel()
                try:
                    await consultation_state["feedback_task"]
                except asyncio.CancelledError:
                    pass
                logger.info("[SCHEDULE] Feedback task cancelado")
            
            consultation_state.update({
                "is_active": False,
                "current_query": None,
                "operation_type": None,
                "start_time": None,
                "task": None,
                "feedback_task": None,
                "last_completed": asyncio.get_event_loop().time(),
                "feedback_sent": []
            })
            
            logger.info("[SCHEDULE] Estado de agendamiento limpio")

    @function_tool()
    async def get_farewell_message(self, ctx: RunContext):
        """Obtiene el mensaje de despedida apropiado según la hora"""
        farewell = get_farewell()
        msg = f"{farewell}. Gracias por contactar a Multiservicioscall."
        return msg

    @function_tool()
    async def handle_user_greeting_or_check(
        self,
        ctx: RunContext,
        user_message: Annotated[str, "Mensaje del usuario que parece ser un saludo o verificación de conexión"]
    ) -> str:
        """Maneja saludos y verificaciones de conexión inmediatamente"""
        
        message_lower = user_message.lower().strip()
        
        greetings = ["hola", "hello", "buenas", "buenos dias", "buenas tardes", "buenas noches"]
        checks = ["¿hola?", "hola?", "estás ahí", "estas ahi", "me escuchas", "sí, es ahí", "si, es ahi"]
        
        if any(greeting in message_lower for greeting in greetings):
            return "¡Hola! Soy Mia de Multiservicioscall. ¿En qué puedo ayudarte?"
        
        if any(check in message_lower for check in checks):
            return "Sí, estoy aquí y te escucho perfectamente. ¿Cómo puedo asistirte?"
            
        return "Te escucho. ¿Podrías repetir tu consulta para ayudarte mejor?"

    @function_tool()
    async def handle_user_impatience(
        self, 
        ctx: RunContext,
        user_message: Annotated[str, "Mensaje del usuario expresando impaciencia o preguntando por demoras"]
    ) -> str:
        """Maneja cuando el usuario expresa impaciencia o pregunta por demoras"""
        
        if consultation_state["is_active"]:
            elapsed_time = asyncio.get_event_loop().time() - consultation_state["start_time"]
            
            if consultation_state["operation_type"] == "appointment":
                if elapsed_time < 5:
                    return "Disculpa, estoy procesando tu cita. Te respondo en unos segundos."
                elif elapsed_time < 10:
                    return "Te pido paciencia, estoy confirmando la disponibilidad del gerente."
                else:
                    return "Entiendo tu impaciencia, la cita está casi lista. Te confirmo en breve."
            else:
                if elapsed_time < 5:
                    return "Disculpa, estoy procesando tu consulta. Te respondo en unos segundos más."
                elif elapsed_time < 10:
                    return "Te pido paciencia, estoy buscando la información más precisa para ti."
                else:
                    return "Entiendo tu impaciencia, la consulta está tomando más tiempo del esperado. Te aseguro que tendrás la respuesta muy pronto."
        else:
            return "¿En qué puedo ayudarte? Estoy aquí para resolver tus dudas sobre Multiservicioscall."

def prewarm(proc: JobProcess):
    """Precarga el modelo VAD para detectar voz"""
    vad = silero.VAD.load()
    proc.userdata["vad"] = vad

class FixedConversationManager:
    """
    🔧 GESTOR CORREGIDO: Evita interferir con consultas activas y reduce interrupciones
    """
    def __init__(self, session: AgentSession):
        self.session = session
        self.last_user_activity = asyncio.get_event_loop().time()
        self.last_agent_response = None
        self.silence_check_task = None
        self.silence_warnings = 0
        self.max_silence_warnings = 1
        self.is_active = True
        
    async def start_monitoring(self):
        """Inicia el monitoreo de silencios corregido"""
        logger.info("[SILENCE] Iniciando monitoreo corregido")
        self.silence_check_task = asyncio.create_task(self._fixed_silence_monitor())
        
    async def stop_monitoring(self):
        """Detiene el monitoreo"""
        logger.info("[SILENCE] Deteniendo monitoreo")
        self.is_active = False
        if self.silence_check_task:
            self.silence_check_task.cancel()
            try:
                await self.silence_check_task
            except asyncio.CancelledError:
                pass
                
    def update_user_activity(self):
        """Actualiza actividad del usuario"""
        self.last_user_activity = asyncio.get_event_loop().time()
        self.silence_warnings = 0
        logger.debug("[SILENCE] Actividad actualizada")
        
    def mark_agent_response(self):
        """Marca cuándo el agente respondió"""
        self.last_agent_response = asyncio.get_event_loop().time()
        logger.debug("[SILENCE] Respuesta del agente marcada")
        
    async def _fixed_silence_monitor(self):
        """
        🔧 MONITOR CORREGIDO: 
        - NUNCA interrumpe durante consultas activas
        - Espera más tiempo después de consultas completadas
        - Solo envía mensajes cuando realmente es apropiado
        """
        while self.is_active:
            try:
                await asyncio.sleep(30)
                
                if not self.is_active:
                    break
                    
                current_time = asyncio.get_event_loop().time()
                silence_duration = current_time - self.last_user_activity
                
                if consultation_state["is_active"]:
                    logger.debug(f"[SILENCE] {consultation_state['operation_type'].capitalize() if consultation_state['operation_type'] else 'Operación'} activo - NO interrumpiendo (silencio: {silence_duration:.1f}s)")
                    continue
                
                if consultation_state.get("last_completed"):
                    time_since_completion = current_time - consultation_state["last_completed"]
                    if time_since_completion < 45:
                        logger.debug(f"[SILENCE] Operación recién completada - dando tiempo extra (completada hace {time_since_completion:.1f}s)")
                        continue
                
                logger.info(f"[SILENCE] Tiempo de silencio: {silence_duration:.1f}s")
                
                should_check_in = (
                    silence_duration >= 60 and
                    self.silence_warnings < self.max_silence_warnings
                )
                
                if should_check_in:
                    self.silence_warnings += 1
                    logger.info(f"[SILENCE] Enviando verificación única después de {silence_duration:.1f}s")
                    
                    try:
                        await self.session.say("¿Hay algo más en lo que pueda ayudarte?")
                        self.mark_agent_response()
                    except Exception as e:
                        logger.error(f"[SILENCE] Error al enviar mensaje: {e}")
                        
                    self.last_user_activity = current_time
                    
                elif silence_duration >= 300:
                    logger.info("[SILENCE] Periodo muy largo - modo completamente pasivo")
                    self.silence_warnings = 0
                    self.last_user_activity = current_time
                    
            except asyncio.CancelledError:
                logger.info("[SILENCE] Monitor cancelado")
                break
            except Exception as e:
                logger.error(f"[SILENCE] Error en monitor: {e}")
                await asyncio.sleep(30)

async def entrypoint(ctx: JobContext):
    """Función principal con manejo corregido"""
    logger.info(f"[ENTRYPOINT] Nueva llamada en {ctx.room.name}")
    global active_session, current_participant_info
    
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        participant = await ctx.wait_for_participant()
        logger.info(f"[ENTRYPOINT] Participante: {participant.identity}")

        current_participant_info = {
            "participant_id": participant.identity,
            "phone_number": extract_phone_from_identity(participant.identity),
            "session_id": ctx.room.name
        }

        assistant = Assistant()
        usage_collector = metrics.UsageCollector()

        def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
            metrics.log_metrics(agent_metrics)
            usage_collector.collect(agent_metrics)

        session = AgentSession(
            vad=ctx.proc.userdata["vad"],
            min_endpointing_delay=1.5,
            max_endpointing_delay=10.0,
        )
        active_session = session

        conversation_manager = FixedConversationManager(session)

        def on_user_speech_committed(event):
            logger.debug("[EVENT] Usuario terminó de hablar")
            conversation_manager.update_user_activity()

        def on_user_started_speaking(event):
            logger.debug("[EVENT] Usuario comenzó a hablar")
            conversation_manager.update_user_activity()

        def on_agent_speech_committed(event):
            logger.debug("[EVENT] Agente terminó de responder")
            conversation_manager.mark_agent_response()

        session.on("user_speech_committed", on_user_speech_committed)
        session.on("user_started_speaking", on_user_started_speaking)
        session.on("agent_speech_committed", on_agent_speech_committed)
        session.on("metrics_collected", on_metrics_collected)

        await session.start(
            room=ctx.room,
            agent=assistant,
            room_input_options=RoomInputOptions(),
        )

        await asyncio.sleep(2)
        greeting = get_greeting()
        welcome_message = f"{greeting}, bienvenido a Multiservicioscall, soy Mia. ¿Cómo puedo ayudarte?"
        await session.say(welcome_message)
        conversation_manager.mark_agent_response()

        await conversation_manager.start_monitoring()

        logger.info("[SESSION] Sesión iniciada con manejo corregido")
        await asyncio.Future()
        
    except Exception as e:
        logger.error(f"[ERROR] Error en sesión: {e}", exc_info=True)
    finally:
        logger.info("[SESSION] Cerrando sesión")
        if 'conversation_manager' in locals():
            await conversation_manager.stop_monitoring()
        
        consultation_state.update({
            "is_active": False,
            "current_query": None,
            "operation_type": None,
            "task": None,
            "feedback_task": None,
            "start_time": None,
            "feedback_sent": []
        })
        active_session = None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not os.getenv("N8N_RAG_WEBHOOK_URL") or not os.getenv("N8N_APPOINTMENT_WEBHOOK_URL"):
        logger.error("[MAIN] N8N_RAG_WEBHOOK_URL o N8N_APPOINTMENT_WEBHOOK_URL requeridas")
        exit(1)
    
    logger.info("[MAIN] Iniciando agente de voz corregido...")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )