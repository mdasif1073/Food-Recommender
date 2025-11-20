from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import datetime
import uuid

def iso(dt: Optional[datetime.datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None

@dataclass
class User:
    user_id: str
    email: str
    password_hash: str
    username: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    liked_foods: List[str] = field(default_factory=list)
    disliked_foods: List[str] = field(default_factory=list)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    last_active: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    @staticmethod
    def new(email: str, password_hash: str, username: Optional[str] = None) -> "User":
        return User(user_id=str(uuid.uuid4()), email=email, password_hash=password_hash, username=username)

    def to_dict(self):
        d = asdict(self)
        d["created_at"] = iso(self.created_at)
        d["last_active"] = iso(self.last_active)
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "User":
        data = dict(data)
        data.pop("_id", None)
        return User(**data)

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
    spice_level: Optional[str] = ""
    cuisine: Optional[str] = ""
    area: Optional[str] = ""

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_payload(payload: Dict[str, Any]) -> "Food":
        payload = dict(payload)
        payload.pop("_id", None)
        return Food(**payload)

@dataclass
class Restaurant:
    restaurant_id: str
    restaurant_name: str
    address: Optional[str] = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    cuisine_types: Optional[str] = ""
    avg_rating: Optional[float] = None
    opening_hours: Optional[str] = ""
    contact_number: Optional[str] = ""
    delivery_available: Optional[bool] = False
    dine_in_available: Optional[bool] = True
    features: Optional[str] = ""
    restaurant_type: Optional[str] = ""
    price_level: Optional[str] = ""
    area: Optional[str] = ""

    def to_dict(self):
        return asdict(self)

@dataclass
class Session:
    session_id: str
    user_id: str
    dialog_history: List[Dict[str, Any]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    last_activity: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    def to_dict(self):
        d = asdict(self)
        d["last_activity"] = iso(self.last_activity)
        return d

@dataclass
class Feedback:
    user_id: str
    food_id: Optional[str] = None
    restaurant_id: Optional[str] = None
    action: str = "like"
    comment: Optional[str] = None
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    def to_dict(self):
        d = asdict(self)
        d["timestamp"] = iso(self.timestamp)
        return d