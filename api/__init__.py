from .auth import router as auth_router
from .upload import router as upload_router
from .search import router as search_router
from .view_image import router as view_router

# List of routers to be included in main
routers = [auth_router, upload_router, search_router , view_router]
