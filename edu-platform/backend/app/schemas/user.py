from pydantic import BaseModel, ConfigDict, EmailStr


class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    role: str


class UserCreate(UserBase):
    password: str


class UserRead(UserBase):
    id: str
    is_active: bool

    model_config = ConfigDict(from_attributes=True)
