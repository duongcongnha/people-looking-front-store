from infrastructure.database.base import session_factory
from infrastructure.database.Person import Person
from sqlalchemy import and_
from datetime import datetime
from typing import List

def add_person_to_db(person:Person):
    session = session_factory()
    session.add(person)
    session.commit()
    session.close() 

def get_all_persons():
    session = session_factory()
    people_query = session.query(Person)
    session.close()
    return people_query.all()

def list_Persons_to_list_dict(all_persons:List[Person]):
    result = [x.to_dict() for x in all_persons]
    return result

def get_persons_in_timerange(start_datetime:datetime, end_datetime:datetime):
    session = session_factory()
    people_query = session.query(Person).filter(and_(Person.in_time >= start_datetime,\
                                                     Person.in_time <= end_datetime))
    session.close()
    all_persons = people_query.all()
    all_persons = list_Persons_to_list_dict(all_persons)
    return all_persons
