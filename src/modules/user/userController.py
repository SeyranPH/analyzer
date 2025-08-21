from fastapi import APIRouter, status
from src.modules.user.userService import addUser, getUser, getUsers, updateUser, deleteUser
from src.modules.user.userModel import UserCreate, UserResponse, UserUpdate

userRouter = APIRouter(prefix="/user", tags=["user"])

@userRouter.post("/users/", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def create_user(user: UserCreate):
    return await addUser(user)

@userRouter.get("/user/{id}", status_code=status.HTTP_200_OK, response_model=UserResponse)
async def get_user(id: int):
    return await getUser(id)

@userRouter.get("/users/", status_code=status.HTTP_200_OK, response_model=list[UserResponse])
async def get_users(limit: int = 10, offset: int = 0):
    return await getUsers(limit, offset)

@userRouter.put("/user/{id}", status_code=status.HTTP_200_OK, response_model=UserResponse)
async def update_user(id: int, user: UserUpdate):
    return await updateUser(id, user)

@userRouter.delete("/user/{id}", status_code=status.HTTP_200_OK)
async def delete_user(id: int):
    return await deleteUser(id)