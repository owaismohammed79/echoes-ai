# Echoes Server

This is the server component for the Echoes walking companion application.

## Demo Mode Implementation

The demo mode is activated when the daily spend cap is exceeded. The system:
1. Tracks spend using `SpendLedger` in `app/services/budgets.py`
2. Activates demo mode when `ledger.exhausted` is True
3. Returns demo responses with appropriate headers

## Key Components

- **Budget Tracking**: `app/services/budgets.py` implements daily spend limits
- **Demo Mode**: Activated via `demo_mode()` property in `Guardrails` class
- **API Integration**: `app/api/limits.py` handles admission decisions
- **Health Endpoint**: `/api/health` reports demo status

## Testing Demo Mode

Run the demo mode tests:
```bash
cd server
python -m pytest tests/test_limits.py::test_exhausting_the_cap_serves_the_demo_without_calling_anyone -v
```

## Environment Configuration

Set these environment variables to control demo mode:
- `ECHOES_DAILY_SPEND_CAP_USD` - Daily spend limit (default: 5.00)
- `ECHOES_FORCE_DEMO_MODE` - Force demo mode (default: false)