from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base


class ClassificationLog(Base):
    __tablename__ = "classification_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    predicted_class= Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    ip_request_rate = Column(Float, nullable=True)
    is_exploit_path = Column(Integer, nullable=True)
    is_known_scanner = Column(Integer, nullable=True)
    payload_size = Column(Float, nullable=True)

