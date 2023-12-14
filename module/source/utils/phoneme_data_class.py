import numpy as np
import librosa
import librosa.display
import os
import re
from sklearn.utils import resample
import random 

__all__ = [
    "Segment",
    "SegmentationLabel",
]
class Segment:
    """
    a unit of speech (i.e. phoneme, mora)
    """
    def __init__(self, tStart, tEnd, label):
        self.tStart = tStart
        self.tEnd = tEnd
        self.label = label

    def __add__(self, other):
        return Segment(self.tStart, other.tEnd, self.label + other.label)

    def can_follow(self, other):
        """
        return True if Segment self can follow Segment other in one mora,
        otherwise return False
        example: (other, self)
             True: ('s', 'a'), ('sh', 'i'), ('ky', 'o:'), ('t', 's')
             False: ('a', 'q'), ('a', 's'), ('u', 'e'), ('s', 'ha')
        """
        vowels = ['a', 'i', 'u', 'e', 'o', 'a:', 'i:', 'u:', 'e:', 'o:']
        consonants = ['w', 'r', 't', 'y', 'p', 's', 'd', 'f', 'g', 'h', 'j',
                      'k', 'z', 'c', 'b', 'n', 'm']
        only_consonants = lambda x: all([c in consonants for c in x])
        if only_consonants(other.label) and self.label in vowels:
            return True
        if only_consonants(other.label) and only_consonants(self.label):
            return True
        return False


class SegmentationLabel:
    """
    list of segments
    """
    def __init__(self, segments, separatedByMora=False):
        self.segments = segments
        self.separatedByMora = separatedByMora
    def by_moras(self):
        """
        return new SegmentationLabel object whose segment are moras
        """
        if self.separatedByMora == True:
            return self

        moraSegments = []
        curMoraSegment = None
        for segment in self.segments:
            if curMoraSegment is None:
                curMoraSegment = segment
            elif segment.can_follow(curMoraSegment):
                curMoraSegment += segment
            else:
                moraSegments.append(curMoraSegment)
                curMoraSegment = segment
        if curMoraSegment:
            moraSegments.append(curMoraSegment)
        return SegmentationLabel(moraSegments, separatedByMora=True)