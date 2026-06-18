import asyncio
import logging
import discord

logger = logging.getLogger("ui")

class CancelView(discord.ui.View):
    """LLM思考・返答生成処理を中断するためのキャンセルボタン付きDiscord View"""
    def __init__(self, task: asyncio.Task, status_msg: discord.Message):
        super().__init__(timeout=60.0)
        self.task = task
        self.status_msg = status_msg

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.danger, emoji="🛑")
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.warning(f"Message processing for channel {interaction.channel_id} cancelled by user.")
        self.task.cancel()
        self.disable_all_items()
        
        embed = discord.Embed(
            title="Cancelled 🛑",
            description="処理がユーザーによって中断されました。",
            color=0xe74c3c
        )
        try:
            await interaction.response.edit_message(embed=embed, view=None)
        except Exception:
            pass

        await asyncio.sleep(3)
        try:
            await self.status_msg.delete()
        except discord.errors.NotFound:
            pass
        except Exception as e:
            logger.error(f"Failed to delete status message: {e}")

def create_status_embed(title: str, description: str, color: int = 0x3498db) -> discord.Embed:
    """進捗ステータス用の標準Embedを作成する"""
    return discord.Embed(
        title=title,
        description=description,
        color=color
    )
