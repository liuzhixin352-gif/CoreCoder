from corecoder.tracing import (
    InMemoryTracer,
    TraceEvent,
    Tracer,
)


def test_in_memory_tracer_records_structured_event():
    tracer = InMemoryTracer()

    event = TraceEvent(
        name="tool.started",
        timestamp=123.0,
        attributes={
            "tool_name": "read_repository_file",
        },
    )

    tracer.emit(event)

    assert tracer.events == [event]

def test_in_memory_tracer_implements_tracer():
    tracer = InMemoryTracer()

    assert isinstance(tracer, Tracer)

def test_in_memory_tracer_filters_events_by_run_id():
    tracer = InMemoryTracer()

    tracer.emit(
        TraceEvent(
            name="agent.started",
            timestamp=1.0,
            attributes={"run_id": "run-1"},
        )
    )
    tracer.emit(
        TraceEvent(
            name="agent.started",
            timestamp=2.0,
            attributes={"run_id": "run-2"},
        )
    )
    tracer.emit(
        TraceEvent(
            name="llm.completed",
            timestamp=3.0,
            attributes={"run_id": "run-1"},
        )
    )

    events = tracer.events_for_run("run-1")

    assert [event.name for event in events] == [
        "agent.started",
        "llm.completed",
    ]