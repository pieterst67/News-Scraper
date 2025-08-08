# news_scraper
Daily news scraper with smart clustering bringin the most important news in your mailbox

## Installation

```
pip install -r requirements.txt
```

## `.env` variables

```
OPENAI_API_KEY=<your-key>
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=user@example.com
SMTP_PASS=<password>
EMAIL_FROM=news-bot@example.com
EMAIL_TO=user@example.com
```

## Config variables
`FEEDS`\
list of RSS/Atom URLs

`READ_LIMIT_WORDS`\
max words per digest (~200 wpm × 15 min = 3000)
