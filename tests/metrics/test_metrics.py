from src.metrics.metrics import API_LATENCY, API_CALLS

def test_api_latency_metric():
    assert API_LATENCY._name == "source_api_latency_seconds"
    assert API_LATENCY._documentation == "Source API latency in seconds by source, source_id, method and status"
    assert API_LATENCY._labelnames == ("source", "source_id", "method", "status")

def test_api_calls_metric():
    assert API_CALLS._name == "source_api_calls"
    assert API_CALLS._documentation == "Source API call count by source, source_id, method and status"
    assert API_CALLS._labelnames == ("source", "source_id", "method", "status")
