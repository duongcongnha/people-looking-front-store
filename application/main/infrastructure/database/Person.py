from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Interval
from sqlalchemy.ext.declarative import declarative_base
from infrastructure.database.base import Base

class Person(Base):
    __tablename__ = 'person'
    # Here we define columns for the table person
    # Notice that each column is also a normal Python instance attribute.
    ID = Column(String, primary_key=True)
    in_time = Column(DateTime)
    exit_time = Column(DateTime)
    duration_attention = Column(Interval)
    
    def __init__(self, ID, in_time, exit_time, duration_attention):
        self.ID = ID
        self.in_time = in_time
        self.exit_time = exit_time
        self.duration_attention = duration_attention

    def to_dict(self):
        FMT = "%d-%m-%Y %H:%M:%S"
        person_dict = {}
        person_dict[self.ID] = {}
        person_dict[self.ID]['in_time'] = self.in_time.strftime(FMT)
        person_dict[self.ID]['exit_time'] = self.exit_time.strftime(FMT)
        person_dict[self.ID]['duration_attention'] = str(self.duration_attention).split(".")[0]
        return person_dict