#!/usr/bin/python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class Factory(ABC):
    """The base for any factory class."""

    @abstractmethod
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
