#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from brain import event


@pytest.fixture
def rand():
    return np.random.rand()

def test_verifier(rand):
    e = event.Event()
    verify = event.EventVerifier(e)

    assert not verify.has_run
    assert verify.count == 0
    e(rand)

    assert verify.has_run
    assert verify.count == 1
    assert verify.run_args == [rand]

def test_event_queue_tick(rand):
    queue = event.EventQueue()
    e = event.Event()
    verify = event.EventVerifier(e)

    queue.add(e, rand)
    queue.tick()

    assert verify.has_run
    assert verify.run_args == [rand]

def test_event_queue_add_within_tick(rand):
    queue = event.EventQueue()
    events = [event.Event() for _ in range(3)]
    verifiers = [event.EventVerifier(events[i]) for i in range(3)]

    # Third event will be added when the first event is triggered
    # on the first tick.
    append = lambda x: queue.add(events[2], x)
    events[0].subscribe(append)

    # Add first two events to the queue
    queue.add(events[0], rand)
    queue.add(events[1], -rand)

    # After the first tick, the third event hasn't run, but is in the queue.
    queue.tick()
    assert not verifiers[2].has_run

    # After the second tick, all events have run.
    queue.tick()
    for verifier in verifiers:
        assert verifier.count == 1

    # Value is passed properly
    assert verifiers[2].run_args == [rand]
