# Question/Answering system over SQL data

## How to run app
1. Clone the repo
2. create .env, `cp .env.example .env`
3. create virtual environment `python3 -m venv venv --prompt chat-agent`
4. activate virtual environment `source venv/bin/activate` 
5. install required packages `pip install -r requirements.txt`
6. start asgi server `uvicorn app:app --reload`


## API ENDPOINT
POST
http://127.0.0.1:8000/query

### Request Body:
```
{
  "question": "how many Employee?"
}
``` 
### Request Header 
send key in header name `APIKEY` value you set on .env with key (ACCESS_TOKEN) 


## Response
```json
{
  "response": [
    {
      "agent": {
        "messages": [
          {
            "content": "",
            "additional_kwargs": {
              "tool_calls": [
                {
                  "id": "call_grwxOWafRySooQEqwp6X19O4",
                  "function": {
                    "arguments": "{\"query\":\"SELECT COUNT(*) AS employee_count FROM Employee\"}",
                    "name": "sql_db_query"
                  },
                  "type": "function"
                }
              ]
            },
            "response_metadata": {
              "token_usage": {
                "completion_tokens": 22,
                "prompt_tokens": 668,
                "total_tokens": 690
              },
              "model_name": "gpt-3.5-turbo-0125",
              "system_fingerprint": null,
              "finish_reason": "tool_calls",
              "logprobs": null
            },
            "type": "ai",
            "name": null,
            "id": "run-acac4c94-c0ea-47a2-ae61-be4421b5b53a-0",
            "example": false,
            "tool_calls": [
              {
                "name": "sql_db_query",
                "args": {
                  "query": "SELECT COUNT(*) AS employee_count FROM Employee"
                },
                "id": "call_grwxOWafRySooQEqwp6X19O4"
              }
            ],
            "invalid_tool_calls": [],
            "usage_metadata": {
              "input_tokens": 668,
              "output_tokens": 22,
              "total_tokens": 690
            }
          }
        ]
      }
    },
    {
      "tools": {
        "messages": [
          {
            "content": "[(8,)]",
            "additional_kwargs": {},
            "response_metadata": {},
            "type": "tool",
            "name": "sql_db_query",
            "id": "2868d4e9-3559-488f-8ce7-ac68e8d56ff8",
            "tool_call_id": "call_grwxOWafRySooQEqwp6X19O4"
          }
        ]
      }
    },
    {
      "agent": {
        "messages": [
          {
            "content": "There are 8 employees in the database.",
            "additional_kwargs": {},
            "response_metadata": {
              "token_usage": {
                "completion_tokens": 10,
                "prompt_tokens": 703,
                "total_tokens": 713
              },
              "model_name": "gpt-3.5-turbo-0125",
              "system_fingerprint": null,
              "finish_reason": "stop",
              "logprobs": null
            },
            "type": "ai",
            "name": null,
            "id": "run-c146d26b-4e8b-4dab-8b5f-fb500da85e04-0",
            "example": false,
            "tool_calls": [],
            "invalid_tool_calls": [],
            "usage_metadata": {
              "input_tokens": 703,
              "output_tokens": 10,
              "total_tokens": 713
            }
          }
        ]
      }
    }
  ]
}
```