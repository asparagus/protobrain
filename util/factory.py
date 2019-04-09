#!/usr/bin/python
# -*- coding: utf-8 -*-
import abc


class Factory(abc.ABC):
    """The base for any factory class."""

    @abc.abstractmethod
    def create(self):
        raise NotImplementedError()

    def __call__(self):
        return self.create()

class ConstantFactory(Factory):
    """A Factory class that always returns the same object."""

    def __init__(self, constant):
        """Initializes the factory with the constant object to return."""
        self.constant = constant

    def create(self):
        """Return the constant."""
        return self.constant

class SequenceFactory(Factory):
    """A Factory class that returns objects from a sequence."""

    def __init__(self, sequence):
        """Initializes the factory with the sequence of objects to create."""
        self.sequence = iter(sequence)

    def create(self):
        """Return the next item."""
        return next(self.sequence)
