
from src.modules.user.userModel import User, database, UserCreate, UserResponse, UserUpdate

async def addUser(user: UserCreate):
  query = User.insert().values(name=user.name, email=user.email)
  last_record_id = await database.execute(query)
  return UserResponse(id=last_record_id, name=user.name, email=user.email)

async def getUser(id: int):
  query = User.select().where(User.c.id == id)
  user = await database.fetch_one(query)
  if user:
    return UserResponse(id=user.id, name=user.name, email=user.email)
  return None

async def getUsers(limit: int, offset: int):
  query = User.select().limit(limit).offset(offset)
  users = await database.fetch_all(query)
  return [UserResponse(id=user.id, name=user.name, email=user.email) for user in users]

async def updateUser(id: int, user: UserUpdate):
  update_values = {}
  if user.name is not None:
    update_values["name"] = user.name
  if user.email is not None:
    update_values["email"] = user.email
  
  if update_values:
    query = User.update().where(User.c.id == id).values(**update_values)
    await database.execute(query)
  
  # Return updated user
  return await getUser(id)

async def deleteUser(id: int):
  query = User.delete().where(User.c.id == id)
  await database.execute(query)
  return {"message": "User deleted successfully"}