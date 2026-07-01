import os
import discord
from discord.voice import VoiceClient
from config import DISCORD_BOT_TOKEN, FFMPEG_PATH, OUTPUT_DIR
from llm import generate_response, reset_conversation
from tts import tts


class FriendBot(discord.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        super().__init__(intents=intents)

    async def on_ready(self) -> None:
        print(f"Bot logged in as {self.user} (ID: {self.user.id})")
        tts.load()
        print("------")


bot = FriendBot()


@bot.slash_command(name="ping", description="Проверка работоспособности")
async def ping(ctx: discord.ApplicationContext) -> None:
    await ctx.respond(f"Pong! Latency: {round(bot.latency * 1000)}ms")


@bot.slash_command(name="join", description="Подключить бота к голосовому каналу")
async def join(ctx: discord.ApplicationContext) -> None:
    if not ctx.author.voice:
        await ctx.respond("Ты не в голосовом канале!", ephemeral=True)
        return

    channel = ctx.author.voice.channel
    await channel.connect()
    await ctx.respond(f"Подключился к `{channel.name}`")


@bot.slash_command(name="leave", description="Отключить бота от голосового канала")
async def leave(ctx: discord.ApplicationContext) -> None:
    voice = ctx.voice_client
    if voice is None:
        await ctx.respond("Я и так не в голосовом канале!", ephemeral=True)
        return

    await voice.disconnect()
    await ctx.respond("Отключился")


def _play_audio(voice: VoiceClient, filepath: str) -> None:
    if voice.is_playing():
        voice.stop()
    voice.play(discord.FFmpegPCMAudio(executable=FFMPEG_PATH, source=filepath))


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
