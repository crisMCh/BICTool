"""Application / Orchestration Layer: scheduling, plugin orchestration, workflow coordination.

`worker.Worker` + `scheduler.Scheduler` run Domain-layer calls (currently just
`evaluate_slice_quality`) off the GUI thread via Qt's `QThreadPool`, so the
Presentation layer has a real async contract to build against. `plugin_manager`
is intentionally still a stub - FR-08's plugin system has no second analysis
module driving its design yet.
"""
