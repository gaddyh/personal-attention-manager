


from datetime import datetime
from pydantic import BaseModel

class Message(BaseModel):
    sender: str
    text: str
    sent_time: datetime