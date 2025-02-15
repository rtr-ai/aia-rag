# RTR AI-RAG Project
This is the repository for the RTR AI Act RAG.

## Matomo Configuration

Matomo can be configured using a .env environment variable.

### Environment Configuration

#### Secure Token Management


1. Create an `.env`using matomo.env.example as reference file:
```
MATOMO_ENDPOINT=https://your-matomo-instance.com/matomo.php
MATOMO_TOKEN=your_tracking_token
MATOMO_SITE_ID=your_site_id
```

### Tracking Behavior
- If Matomo configuration is missing, tracking silently skips
- Supports tracking events with optional JSON  or string metadata
- Uses singleton service for consistent tracking across application

### Usage Example
```python
from services.matomo_tracking_service import matomo_service

# Track event with optional data
matomo_service.track_event(
    action='user_prompt', 
    data={'May I use AI in my organization?'}
)
```