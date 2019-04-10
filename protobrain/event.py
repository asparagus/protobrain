#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy


class Event:
    """A class for an event that can be subscribed to."""

    def __init__(self):
        """Initialize the event."""
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def __call__(self, value):
        for subscriber in self.subscribers:
            subscriber(value)


class EventQueue:
    """A queue for holding-off events."""

    def __init__(self):
        """Initialize the queue empty."""
        self._hold_off_events = {}

    def tick(self):
        """Trigger all events in the queue."""
        backup = self._hold_off_events
        self._hold_off_events = {}
        for event in backup:
            event(backup[event])

    def add(self, event, value):
        """
        Add a new event to the queue for the next tick.

        The event will run with the given value.
        """
        self._hold_off_events[event] = value


class EventVerifier:
    """Helper class to verify an event has run."""

    def __init__(self, event):
        """Initialize the EventVerifier."""
        self._event_args = []
        event.subscribe(self._register)

    def _register(self, value):
        """Helper method to increment the counts."""
        self._event_args.append(value)

    @property
    def has_run(self):
        """Whether the event has run."""
        return self.count > 0

    @property
    def count(self):
        """How many times the event has run."""
        return len(self._event_args)

    @property
    def run_args(self):
        """The arguments that have been used when running the event."""
        return copy.deepcopy(self._event_args)

