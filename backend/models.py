from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import datetime
import uuid

@dataclass
class User:
    user_id: str                      # Qdrant/Mongo UUID (NEVER email!)
    email: str                        # For login/account reference only
    password_hash: str
    username: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    liked_foods: List[str] = field(default_factory=list)
    disliked_foods: List[str] = field(default_factory=list)
    chat_history: List[Dict] = field(default_factory=list)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    last_active: Optional[datetime.datetime] = None

    # The KEY part to guarantee UUID is used every time a user is created:
    @staticmethod
    def new(email: str, password_hash: str, username: Optional[str] = None) -> "User":
        user_uuid = str(uuid.uuid4())
        return User(user_id=user_uuid, email=email, password_hash=password_hash, username=username)


    def to_dict(self):
        d = asdict(self)
        d["created_at"] = d.get("created_at", datetime.datetime.now()).isoformat()
        d["last_active"] = d.get("last_active", datetime.datetime.now()).isoformat() if d.get("last_active") else None
        return d


    @staticmethod
    def from_dict(data):
        data = dict(data)
        data.pop('_id', None)
        return User(**data)


    @staticmethod
    def new(email: str, password_hash: str, username: Optional[str] = None) -> "User":
        # Always create a unique UUID per user for all embeddings/Qdrant records/etc
        user_uuid = str(uuid.uuid4())
        return User(user_id=user_uuid, email=email, password_hash=password_hash, username=username)


# --- Food model ---
@dataclass
class Food:
    food_id: str
    food_name: str
    restaurant_id: str
    description: Optional[str] = ""
    category: Optional[str] = ""
    veg_nonveg: Optional[str] = ""
    ingredients: Optional[str] = ""
    dish_type: Optional[str] = ""
    popular_in: Optional[str] = ""
    price_level: Optional[str] = ""
    food_rating: Optional[float] = None


    def to_dict(self):
        return asdict(self)


# --- Restaurant model ---
@dataclass
class Restaurant:
    restaurant_id: str
    restaurant_name: str
    address: Optional[str] = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    cuisine_types: Optional[str] = ""
    avg_rating: Optional[float] = None
    num_reviews: Optional[int] = 0
    opening_hours: Optional[str] = ""
    contact_number: Optional[str] = ""
    delivery_available: Optional[bool] = False
    dine_in_available: Optional[bool] = False
    features: Optional[str] = ""
    restaurant_type: Optional[str] = ""
    area: Optional[str] = ""
    price_level: Optional[str] = ""
    image_url: Optional[str] = ""


    def to_dict(self):
        return asdict(self)


# --- Session (for conversational context/state) ---
@dataclass
class Session:
    session_id: str
    user_id: str
    dialog_history: List[Dict] = field(default_factory=list)  # [{role:'user/bot', content:''}]
    state: Dict[str, Any] = field(default_factory=dict)
    last_activity: datetime.datetime = field(default_factory=datetime.datetime.now)


    def to_dict(self):
        d = asdict(self)
        d["last_activity"] = d.get("last_activity", datetime.datetime.now()).isoformat()
        return d


# --- Feedback structure for analytics/mod ---
@dataclass
class Feedback:
    user_id: str            # Must be UUID (never email)
    food_id: Optional[str] = None
    restaurant_id: Optional[str] = None
    action: str = "like"    # like, dislike, suggest, report
    comment: Optional[str] = None
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


    def to_dict(self):
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d