"""
Relation Type Definitions for Shared and Exclusive Relations
"""

class RelationTypes:
    """Define shared and exclusive relation types"""
    
    SHARED_RELATIONS = [
        "skills",
        "hobbies", 
        "learned_languages",
        "visited_places"
    ]
    
    EXCLUSIVE_RELATIONS = [
        "residence",
        "current_location",
        "health_status"
    ]
    
    @classmethod
    def is_shared(cls, relation):
        """Check if relation is shared type"""
        return relation in cls.SHARED_RELATIONS
    
    @classmethod
    def is_exclusive(cls, relation):
        """Check if relation is exclusive type"""
        return relation in cls.EXCLUSIVE_RELATIONS