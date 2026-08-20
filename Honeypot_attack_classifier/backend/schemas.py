from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class PredictRequest(BaseModel):
    model_name: str = "gradient_boosting" 
    src_ip: str
    timestamp: str
    dst_port: int
    protocol: str
    ttl: Optional[int] = None
    window_size: Optional[int] = None
    tcp_flag: Optional[str] = None
    src_port: int
    http_method: Optional[str]= None
    http_path: Optional[str] = None
    user_agent: Optional[str] = None
    ssh_client: Optional[str] = None
    payload_size: Optional[float] = None
    request_rate_override: Optional[float] = None
   
class PredictResponse(BaseModel):
    model_used: str
    predicted_class: str
    confidence: float

class LogEntry(BaseModel):
    id: int
    timestamp: datetime
    predicted_class: str
    confidence: float
    ip_request_rate: Optional[float] = None
    is_exploit_path: Optional[int] = None
    is_known_scanner: Optional[int] = None 
    payload_size: Optional[float] = None

    class Config:
        from_attributes= True 
