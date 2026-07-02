import os
import asyncio
import discord
from discord.voice import VoiceClient
from config import DISCORD_BOT_TOKEN, FFMPEG_PATH, OUTPUT_DIR
from llm import generate_response, reset_conversation
from tts import tts
from listener import VoiceListener

voice_listener = VoiceListener()


class FriendBot(discord.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        super().__init__(intents=intents)

    async def on_ready(self) -> None:
        print(f"Bot logged in as {self.user} (ID: {self.user.id})")
        print(f"Guilds: {[g.name for g in self.guilds]}")
        await self.sync_commands(force=True)
        for guild in self.guilds:
            await self.sync_commands(guild_ids=[guild.id])
            print(f"Synced commands to guild: {guild.name}")
        print("Commands synced")

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, tts.load)
        loop.run_in_executor(None, voice_listener.load_whisper)
        print("Models loading in background...")
        print("------")


bot = FriendBot()


def _play_audio(voice: VoiceClient, filepath: str) -> None:
    if voice.is_playing():
        voice.stop()
    voice.play(discord.FFmpegPCMAudio(executable=FFMPEG_PATH, source=filepath))


async def _handle_transcription(text: str) -> None:
    """Callback: transcribed speech -> LLM -> TTS -> play in voice."""
    voice = bot.voice_clients[0] if bot.voice_clients else None
    if voice is None:
        print("[Handle] No voice client")
        return

    try:
        reply = await generate_response(text)
    except Exception as e:
        print(f"[Handle] LLM error: {e}")
        return

    if not reply:
        print("[Handle] Empty reply")
        return

    try:
        output_path = await tts.generate(reply)
        _play_audio(voice, output_path)
        print(f"[Handle] Played: {reply!r}")
    except Exception as e:
        print(f"[Handle] TTS error: {e}")


async def _async_transcribe_and_handle() -> None:
    """Run blocking transcription in executor, then LLM -> TTS -> play."""
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, voice_listener.transcribe_blocking)
    if not text:
        return
    print(f"[Sink] Transcription: {text}")
    await _handle_transcription(text)


class PCMStreamSink(discord.sinks.Sink):
    encoding: str = "pcm"
    __sink_listeners__: list = []

    def __init__(self) -> None:
        super().__init__()
        self._count = 0

    def is_opus(self) -> bool:
        return False

    def write(self, data, user) -> None:
        pcm = getattr(data, "pcm", None)
        if pcm and len(pcm) > 0:
            self._count += 1
            if self._count == 1:
                print(f"[Sink] Got first PCM frame: {len(pcm)} bytes from {user}")
            if voice_listener.is_busy():
                return
            ready = voice_listener.feed_pcm(pcm)
            if ready:
                print(f"[Sink] Triggering async transcription...")
                bot.loop.create_task(_async_transcribe_and_handle())


@bot.slash_command(name="ping", description="Проверка работоспособности")
async def ping(ctx: discord.ApplicationContext) -> None:
    await ctx.respond(f"Pong! Latency: {round(bot.latency * 1000)}ms")


@bot.slash_command(name="join", description="Подключить бота к голосовому каналу")
async def join(ctx: discord.ApplicationContext) -> None:
    if not ctx.author.voice:
        await ctx.respond("Ты не в голосовом канале!", ephemeral=True)
        return

    await ctx.defer()
    channel = ctx.author.voice.channel
    await channel.connect()
    await ctx.respond(f"Подключился к `{channel.name}`")


@bot.slash_command(name="leave", description="Отключить бота от голосового канала")
async def leave(ctx: discord.ApplicationContext) -> None:
    voice = ctx.voice_client
    if voice is None:
        await ctx.respond("Я и так не в голосовом канале!", ephemeral=True)
        return

    voice_listener.stop()
    await voice.disconnect()
    await ctx.respond("Отключился")


@bot.slash_command(
    name="listen", description="Начать слушать голосовой канал и отвечать голосом"
)
async def listen(ctx: discord.ApplicationContext) -> None:
    voice = ctx.voice_client
    if voice is None:
        await ctx.respond("Сначала вызови /join!", ephemeral=True)
        return

    voice.start_listening(PCMStreamSink())
    voice_listener.on_transcription = _handle_transcription
    await ctx.respond("Слушаю голосовой канал... Говори!")


@bot.slash_command(
    name="talk",
    description="Подключиться и начать голосовой диалог (одна команда)",
)
async def talk(ctx: discord.ApplicationContext) -> None:
    if not ctx.author.voice:
        await ctx.respond("Ты не в голосовом канале!", ephemeral=True)
        return

    await ctx.defer()

    channel = ctx.author.voice.channel
    voice = await channel.connect()

    voice.start_listening(PCMStreamSink())
    voice_listener.on_transcription = _handle_transcription

    await ctx.respond(
        f"Подключился к `{channel.name}` и слушаю. Говори!"
    )


@bot.slash_command(name="stoplisten", description="Перестать слушать канал")
async def stoplisten(ctx: discord.ApplicationContext) -> None:
    voice = ctx.voice_client
    if voice is None:
        await ctx.respond("Я и так не в голосовом канале!", ephemeral=True)
        return

    voice.stop_listening()
    voice_listener.stop()
    await ctx.respond("Перестал слушать")


@bot.slash_command(name="chat", description="Написать боту — он ответит голосом")
async def chat(ctx: discord.ApplicationContext, text: str) -> None:
    voice = ctx.voice_client
    if voice is None:
        await ctx.respond("Сначала вызови /join!", ephemeral=True)
        return

    await ctx.defer()

    try:
        reply = await generate_response(text)
    except Exception as e:
        await ctx.respond(f"Ошибка LLM: {e}", ephemeral=True)
        return

    output_path = await tts.generate(reply)
    _play_audio(voice, output_path)
    await ctx.respond(f"> {text}\n{reply}")


@bot.slash_command(name="ask", description="Спросить бота — текстовый ответ")
async def ask(ctx: discord.ApplicationContext, text: str) -> None:
    await ctx.defer()

    try:
        reply = await generate_response(text)
    except Exception as e:
        await ctx.respond(f"Ошибка LLM: {e}", ephemeral=True)
        return

    await ctx.respond(reply)


@bot.slash_command(name="reset", description="Сбросить историю диалога")
async def reset(ctx: discord.ApplicationContext) -> None:
    reset_conversation()
    await ctx.respond("История диалога очищена", ephemeral=True)


@bot.slash_command(name="play", description="Проиграть wav-файл в голосовой канал")
async def play(ctx: discord.ApplicationContext, filepath: str) -> None:
    voice = ctx.voice_client
    if voice is None:
        await ctx.respond("Сначала вызови /join!", ephemeral=True)
        return

    if not os.path.exists(filepath):
        await ctx.respond(f"Файл не найден: {filepath}", ephemeral=True)
        return

    _play_audio(voice, filepath)
    await ctx.respond(f"Проигрываю: {filepath}")


if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN не найден. Добавь его в файл .env")

    bot.run(DISCORD_BOT_TOKEN)
