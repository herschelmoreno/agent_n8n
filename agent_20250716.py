import os
import asyncio
import logging
import httpx
from datetime import datetime
import pytz

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
#comentario
from livekit.plugins import (
    cartesia,     # ⬅️ TTS usado para hablar
    openai,       # ⬅️ LLM usado para generar respuestas
    deepgram,     # ⬅️ STT para transcripción del usuario
    silero,        # ⬅️ Detector de voz/silencio (VAD)
    elevenlabs     # TTS de mejor calidad y personalizando la voz.
)

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

# 🕐 Funciones para saludos dinámicos
def get_current_time_peru():
    """Obtiene la hora actual en Perú"""
    peru_tz = pytz.timezone('America/Lima')
    return datetime.now(peru_tz)

def get_greeting():
    """Retorna el saludo apropiado según la hora"""
    current_time = get_current_time_peru()
    hour = current_time.hour
    
    if 5 <= hour < 12:
        return "Buenos días"
    elif 12 <= hour < 19:
        return "Buenas tardes"
    else:
        return "Buenas noches"

def get_farewell():
    """Retorna la despedida apropiada según la hora"""
    current_time = get_current_time_peru()
    hour = current_time.hour
    
    if 5 <= hour < 12:
        return "Que tenga un buen día"
    elif 12 <= hour < 19:
        return "Que tenga una buena tarde"
    else:
        return "Que tenga una buena noche"

# 🔊 Configurar TTS de ElevenLabs
# Configure TTS
elevenlabs_tts = elevenlabs.TTS(
    voice_id="VywzfvxxNk4yFAaoMm4Q",  # Your voice ID
    #name= "Daniela - Young and Talkative",
    model="eleven_turbo_v2_5",
    #category= "professional",
    api_key=os.getenv("ELEVEN_API_KEY")  # Make sure this is set in .env
)

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Eres un asistente de voz creado por Multiservicioscall."
                "La conversación tiene que ser respetuosa y cortes."
                "Tu interfaz con los usuarios será por voz."
                "Use short, clear responses and avoid unpronounceable punctuation."
                "Cuando respondas los montos de deudas como por ejemplo 250.25 di 'doscientos cincuenta soles con veinticinco centavos'."
                "Si el cliente se queda callado por más de 5 segundos, mantén una actitud proactiva y pregunta si necesita ayuda."
                "Mantén la conversación activa y amigable."
                "Cuando te despidas, usa las funciones de despedida apropiadas según la hora."
                "No te despidas automáticamente a menos que el cliente lo solicite explícitamente."
                "Solo responde sobre las tool y sobre call center."
            ),
            stt=deepgram.STT(model="nova", language="es-419"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(
                model="sonic-2",
                language="es",
                voice="5c5ad5e7-1020-476b-8b91-fdcbe9cc313c"
            ),
            #tts=elevenlabs_tts,  # ⬅️ Ahora usa ElevenLabs
        )

        logger.info(f"[TOOLS] Registradas: {[t.__name__ for t in self._tools]}")

    @function_tool()
    async def lookup_weather(self, ctx: RunContext, city: Annotated[str, "Ciudad para saber el clima"]):
        logger.info(f"[TOOL] Ejecutando lookup_weather para: {city}")
        api_key = os.getenv("OPENWEATHER_API_KEY")

        if not api_key:
            msg = "No se encontró la clave API para el clima."
            return msg

        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&lang=es&units=metric"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                data = response.json()

                if response.status_code != 200:
                    msg = f"No se pudo obtener el clima para {city}. {data.get('message', '')}"
                    return msg

                clima = data["weather"][0]["description"]
                temp = data["main"]["temp"]
                msg = f"El clima en {city} es '{clima}' con una temperatura de {temp}°C."
                return msg

        except Exception as e:
            msg = f"Ocurrió un error al obtener el clima: {str(e)}"
            return msg

    @function_tool()
    async def lookup_debt_info(self, ctx: RunContext, phone_number: Annotated[str, "Número de teléfono del cliente para consultar su deuda"]):
        logger.info(f"[TOOL] Ejecutando lookup_debt_info para: {phone_number}")

        N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

        if not N8N_WEBHOOK_URL:
            msg = "Error: URL del webhook de n8n no configurada."
            return msg

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(N8N_WEBHOOK_URL, json={"phone_number": phone_number})
                response.raise_for_status()

                data = response.json()

                if data.get("status") == "success":
                    debt_info = data.get("debt_info", "No se encontró información de deuda.")
                    msg = f"Aquí está la información de deuda: {debt_info}"
                    return msg
                else:
                    error_msg = data.get("message", "Error desconocido al consultar la deuda.")
                    msg = f"No se pudo obtener la información de deuda: {error_msg}"
                    return msg

        except httpx.RequestError as e:
            msg = f"Error de red al conectar con n8n: {str(e)}"
            return msg
        except httpx.HTTPStatusError as e:
            msg = f"Error en la respuesta de n8n (código {e.response.status_code}): {e.response.text}"
            return msg
        except Exception as e:
            msg = f"Ocurrió un error inesperado: {str(e)}"
            return msg

    @function_tool()
    async def get_farewell_message(self, ctx: RunContext):
        """Obtiene el mensaje de despedida apropiado según la hora"""
        farewell = get_farewell()
        msg = f"{farewell}. Gracias por contactar a Multiservicioscall."
        return msg

    @function_tool()
    async def check_if_user_still_there(self, ctx: RunContext):
        """Función para verificar si el usuario sigue en la conversación"""
        messages = [
            "¿Sigues ahí? ¿Necesitas ayuda con algo más?",
            "¿Todo está bien? Estoy aquí para ayudarte.",
            "¿Hay algo en lo que pueda asistirte?",
            "Si necesitas un momento para revisar la información, no hay problema. Estoy aquí cuando estés listo.",
            "¿Te gustaría que repita alguna información?"
        ]
        import random
        return random.choice(messages)

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

class ConversationManager:
    def __init__(self, session: AgentSession):
        self.session = session
        self.last_user_activity = asyncio.get_event_loop().time()
        self.silence_check_task = None
        self.silence_warnings = 0
        self.max_silence_warnings = 3
        self.is_active = True
        
    async def start_monitoring(self):
        """Inicia el monitoreo de silencios"""
        logger.info("[SILENCE] Iniciando monitoreo de silencios")
        self.silence_check_task = asyncio.create_task(self._monitor_silence())
        
    async def stop_monitoring(self):
        """Detiene el monitoreo de silencios"""
        logger.info("[SILENCE] Deteniendo monitoreo de silencios")
        self.is_active = False
        if self.silence_check_task:
            self.silence_check_task.cancel()
            try:
                await self.silence_check_task
            except asyncio.CancelledError:
                pass
                
    def update_user_activity(self):
        """Actualiza el timestamp de la última actividad del usuario"""
        self.last_user_activity = asyncio.get_event_loop().time()
        self.silence_warnings = 0
        logger.info("[SILENCE] Actividad del usuario detectada, reiniciando contador")
        
    async def _monitor_silence(self):
        """Monitorea los silencios y actúa en consecuencia"""
        while self.is_active:
            try:
                await asyncio.sleep(15)  # Revisa cada 15 segundos
                
                if not self.is_active:
                    break
                    
                current_time = asyncio.get_event_loop().time()
                silence_duration = current_time - self.last_user_activity
                
                logger.info(f"[SILENCE] Tiempo de silencio: {silence_duration:.1f}s")
                
                # Si han pasado más de 15 segundos sin actividad
                if silence_duration >= 15 and self.silence_warnings < self.max_silence_warnings:
                    self.silence_warnings += 1
                    logger.info(f"[SILENCE] Enviando mensaje de verificación #{self.silence_warnings}")
                    
                    messages = [
                        "¿Sigues ahí? ¿Necesitas ayuda con algo más?",
                        "¿Todo está bien? Estoy aquí para ayudarte.",
                        "¿Hay algo en lo que pueda asistirte?",
                        "Si necesitas un momento para revisar la información, no hay problema. Estoy aquí cuando estés listo.",
                        "¿Te gustaría que repita alguna información?"
                    ]
                    
                    message_index = min(self.silence_warnings - 1, len(messages) - 1)
                    try:
                        await self.session.say(messages[message_index])
                    except Exception as e:
                        logger.error(f"[SILENCE] Error al enviar mensaje: {e}")
                        
                    # Resetea el contador después de enviar el mensaje
                    self.last_user_activity = current_time
                    
                # Si han pasado muchos intentos y sigue sin respuesta
                elif silence_duration >= 60 and self.silence_warnings >= self.max_silence_warnings:
                    logger.info("[SILENCE] Tiempo límite alcanzado, manteniendo sesión activa")
                    try:
                        await self.session.say("Estaré aquí cuando necesites ayuda. Puedes hablar cuando gustes.")
                    except Exception as e:
                        logger.error(f"[SILENCE] Error al enviar mensaje final: {e}")
                    
                    # Resetea para seguir monitoreando
                    self.silence_warnings = 0
                    self.last_user_activity = current_time
                    
            except asyncio.CancelledError:
                logger.info("[SILENCE] Monitoreo cancelado")
                break
            except Exception as e:
                logger.error(f"[SILENCE] Error en monitoreo: {e}")
                await asyncio.sleep(5)  # Espera antes de reintentar

async def entrypoint(ctx: JobContext):
    logger.info(f"Conectando a la sala {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"Iniciando asistente para {participant.identity}")

    assistant = Assistant()
    usage_collector = metrics.UsageCollector()

    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        min_endpointing_delay=0.7,
        max_endpointing_delay=4.0,
    )

    # 🔄 Crea el gestor de conversación
    conversation_manager = ConversationManager(session)

    # Registra eventos para detectar actividad del usuario
    def on_user_speech_committed(event):
        logger.info("[EVENT] Usuario habló, actualizando actividad")
        conversation_manager.update_user_activity()

    def on_user_started_speaking(event):
        logger.info("[EVENT] Usuario comenzó a hablar")
        conversation_manager.update_user_activity()

    # Registra los eventos
    session.on("user_speech_committed", on_user_speech_committed)
    session.on("user_started_speaking", on_user_started_speaking)
    session.on("metrics_collected", on_metrics_collected)

    try:
        await session.start(
            room=ctx.room,
            agent=assistant,
            room_input_options=RoomInputOptions(),
        )

        await asyncio.sleep(2)  # Espera para que el audio esté listo
        
        # 🎯 Saludo dinámico basado en la hora
        greeting = get_greeting()
        welcome_message = f"{greeting}, bienvenido a Multiservicioscall, te saluda Mia. ¿En qué puedo ayudarte hoy?"
        await session.say(welcome_message)

        # 🔄 Inicia el monitoreo de silencios
        await conversation_manager.start_monitoring()

        # Mantiene la sesión activa indefinidamente
        logger.info("[SESSION] Sesión iniciada, esperando interacciones...")
        
        # ✅ CAMBIO CRÍTICO: Usar session.wait_for_disconnect() en lugar de session.aclose()
        #await session.wait_for_disconnect()
        await asyncio.Future()
        
    except Exception as e:
        logger.error(f"[ERROR] Error en la sesión: {e}")
    finally:
        # Limpieza al cerrar
        logger.info("[SESSION] Cerrando sesión y limpiando recursos")
        await conversation_manager.stop_monitoring()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )