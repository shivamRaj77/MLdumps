from pydantic import BaseModel

# class which describe banknotes measurements

class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float