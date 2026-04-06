from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from groq import Groq
import json
import requests
import uvicorn

load_dotenv()

# Etherlink backend (Express) — override with BACKEND_BASE_URL e.g. http://localhost:3000
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:3000").rstrip("/")

# Groq LLM (OpenAI-compatible chat + tool calling). Set GROQ_API_KEY in .env
# Default model: strong tool-use and reasoning; override with GROQ_MODEL if needed.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_groq_key = os.getenv("GROQ_API_KEY")
client: Optional[Groq] = Groq(api_key=_groq_key) if _groq_key else None

app = FastAPI(title="Etherlink AI Agent Builder")

# Tool Definitions
TOOL_DEFINITIONS = {
    "transfer": {
        "name": "transfer",
        "description": "Transfer tokens from one address to another. Requires privateKey, toAddress, amount, and tokenAddress.",
        "parameters": {
            "type": "object",
            "properties": {
                "privateKey": {"type": "string", "description": "Private key of the sender wallet"},
                "toAddress": {"type": "string", "description": "Recipient wallet address"},
                "amount": {"type": "string", "description": "Amount of tokens to transfer"},
                "tokenAddress": {"type": "string", "description": "Contract address of the token"}
            },
            "required": ["privateKey", "toAddress", "amount", "tokenAddress"]
        },
        "endpoint": f"{BACKEND_BASE_URL}/transfer",
        "method": "POST"
    },
    "swap": {
        "name": "swap",
        "description": "Swap one token for another. Requires privateKey, tokenIn, tokenOut, amountIn, and slippageTolerance.",
        "parameters": {
            "type": "object",
            "properties": {
                "privateKey": {"type": "string", "description": "Private key of the wallet"},
                "tokenIn": {"type": "string", "description": "Input token contract address"},
                "tokenOut": {"type": "string", "description": "Output token contract address"},
                "amountIn": {"type": "string", "description": "Amount of input tokens"},
                "slippageTolerance": {"type": "number", "description": "Slippage tolerance percentage"}
            },
            "required": ["privateKey", "tokenIn", "tokenOut", "amountIn", "slippageTolerance"]
        },
        "endpoint": f"{BACKEND_BASE_URL}/swap",
        "method": "POST"
    },
    "get_balance": {
        "name": "get_balance",
        "description": "Get Etherlink Shadownet wallet analytics: native XTZ balance plus ERC-20 positions (via Zerion). Requires wallet address.",
        "parameters": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "Wallet address (0x…) to analyze"}
            },
            "required": ["address"]
        },
        "endpoint": f"{BACKEND_BASE_URL}/api/balance/erc20",
        "method": "POST"
    },
    "deploy_erc20": {
        "name": "deploy_erc20",
        "description": "Deploy a new ERC-20 token on Etherlink Shadownet via TokenFactory. Requires privateKey, name, symbol, decimals, and initialSupply (human-readable supply, not wei).",
        "parameters": {
            "type": "object",
            "properties": {
                "privateKey": {"type": "string", "description": "Private key of the deployer wallet"},
                "name": {"type": "string", "description": "Token name"},
                "symbol": {"type": "string", "description": "Token symbol"},
                "decimals": {"type": "integer", "description": "Token decimals (commonly 18)"},
                "initialSupply": {"type": "number", "description": "Initial supply in human units (e.g. 1000000), not totalSupply"}
            },
            "required": ["privateKey", "name", "symbol", "decimals", "initialSupply"]
        },
        "endpoint": f"{BACKEND_BASE_URL}/deploy-token",
        "method": "POST"
    },
    "deploy_erc721": {
        "name": "deploy_erc721",
        "description": "Deploy a new ERC-721 NFT collection. Requires privateKey, name, and symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "privateKey": {"type": "string", "description": "Private key of the deployer wallet"},
                "name": {"type": "string", "description": "NFT collection name"},
                "symbol": {"type": "string", "description": "NFT collection symbol"}
            },
            "required": ["privateKey", "name", "symbol"]
        },
        "endpoint": f"{BACKEND_BASE_URL}/create-nft-collection",
        "method": "POST"
    },
    "create_dao": {
        "name": "create_dao",
        "description": "Create a new DAO (Decentralized Autonomous Organization). Requires privateKey, name, votingPeriod, and quorumPercentage.",
        "parameters": {
            "type": "object",
            "properties": {
                "privateKey": {"type": "string", "description": "Private key of the DAO creator"},
                "name": {"type": "string", "description": "DAO name"},
                "votingPeriod": {"type": "string", "description": "Voting period in seconds"},
                "quorumPercentage": {"type": "string", "description": "Quorum percentage required for voting"}
            },
            "required": ["privateKey", "name", "votingPeriod", "quorumPercentage"]
        },
        "endpoint": f"{BACKEND_BASE_URL}/create-dao",
        "method": "POST"
    },
    "airdrop": {
        "name": "airdrop",
        "description": "Airdrop tokens to multiple recipients. Requires privateKey, recipients (list of addresses), and amount per recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "privateKey": {"type": "string", "description": "Private key of the sender wallet"},
                "recipients": {"type": "array", "items": {"type": "string"}, "description": "List of recipient wallet addresses"},
                "amount": {"type": "string", "description": "Amount to send to each recipient"}
            },
            "required": ["privateKey", "recipients", "amount"]
        },
        "endpoint": f"{BACKEND_BASE_URL}/airdrop",
        "method": "POST"
    },
    "fetch_price": {
        "name": "fetch_price",
        "description": "Fetch the current price of any cryptocurrency or token. Requires a query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query string for token price (e.g., 'bitcoin current price')"}
            },
            "required": ["query"]
        },
        "endpoint": f"{BACKEND_BASE_URL}/token-price",
        "method": "POST"
    },
    "deposit_yield": {
        "name": "deposit_yield",
        "description": "Create a deposit with yield prediction. Requires privateKey, tokenAddress, depositAmount, and apyPercent.",
        "parameters": {
            "type": "object",
            "properties": {
                "privateKey": {"type": "string", "description": "Private key of the depositor wallet"},
                "tokenAddress": {"type": "string", "description": "Token contract address to deposit"},
                "depositAmount": {"type": "string", "description": "Amount to deposit"},
                "apyPercent": {"type": "number", "description": "Annual Percentage Yield (APY) percentage"}
            },
            "required": ["privateKey", "tokenAddress", "depositAmount", "apyPercent"]
        },
        "endpoint": f"{BACKEND_BASE_URL}/yield",
        "method": "POST"
    },
    "wallet_analytics": {
        "name": "wallet_analytics",
        "description": "Get Etherlink wallet analytics (native XTZ + ERC-20 positions via Zerion). Requires wallet address.",
        "parameters": {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "Wallet address to analyze"}
            },
            "required": ["address"]
        },
        "endpoint": f"{BACKEND_BASE_URL}/api/balance/erc20",
        "method": "POST"
    }
}

# Pydantic Models
class ToolConnection(BaseModel):
    tool: str
    next_tool: Optional[str] = None

class AgentRequest(BaseModel):
    tools: List[ToolConnection]
    user_message: str
    private_key: Optional[str] = None

class AgentResponse(BaseModel):
    agent_response: str
    tool_calls: List[Dict[str, Any]]
    results: List[Dict[str, Any]]

# Helper Functions
def build_system_prompt(tool_connections: List[ToolConnection]) -> str:
    """Build a dynamic system prompt based on connected tools"""
    
    # Extract unique tools
    unique_tools = set()
    tool_flow = {}
    
    for conn in tool_connections:
        unique_tools.add(conn.tool)
        if conn.next_tool:
            unique_tools.add(conn.next_tool)
            tool_flow[conn.tool] = conn.next_tool
    
    # Check if sequential execution exists
    has_sequential = any(conn.next_tool for conn in tool_connections)
    
    system_prompt = """You are an AI agent for Etherlink (EVM on Tezos Smart Rollups). You help users perform operations on Etherlink Shadownet testnet using the tools available to you.

AVAILABLE TOOLS:
"""
    
    for tool_name in unique_tools:
        if tool_name in TOOL_DEFINITIONS:
            tool_def = TOOL_DEFINITIONS[tool_name]
            system_prompt += f"\n- {tool_name}: {tool_def['description']}\n"
    
    if has_sequential:
        system_prompt += "\n\nTOOL EXECUTION FLOW:\n"
        system_prompt += "Some tools are connected in sequence. You MUST execute them in the specified order:\n"
        for tool, next_tool in tool_flow.items():
            system_prompt += f"- After {tool} completes, YOU MUST IMMEDIATELY call {next_tool}\n"
        
        system_prompt += """
SEQUENTIAL EXECUTION INSTRUCTIONS - CRITICAL:
1. When tools are connected sequentially, you MUST execute ALL tools in the chain
2. After completing one tool, IMMEDIATELY proceed to call the next tool in the sequence
3. DO NOT wait for user confirmation between sequential tool calls
4. Execute all sequential tools in ONE conversation turn
5. Only provide a final summary after ALL sequential tools have been completed
6. If you have all the required parameters for the entire sequence, execute all tools immediately
"""
    else:
        system_prompt += """
INSTRUCTIONS:
1. You can perform any of the available operations based on user requests
2. Ask for required parameters if not provided
3. Execute the appropriate tool based on user needs
4. Provide clear results and next steps
"""
    
    system_prompt += """
IMPORTANT RULES:
- Only use the tools that are available to you
- Always ask for required parameters before making tool calls
- Be conversational and helpful
- If a privateKey is needed and provided in the context, use it
- Provide transaction hashes and explorer links when available
- Explain what each operation does in simple terms
- For sequential executions, complete the ENTIRE chain before responding
"""
    
    return system_prompt

def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool by calling its API endpoint"""
    
    if tool_name not in TOOL_DEFINITIONS:
        raise ValueError(f"Unknown tool: {tool_name}")

    parameters = dict(parameters)
    if tool_name == "deploy_erc20":
        if "totalSupply" in parameters and "initialSupply" not in parameters:
            parameters["initialSupply"] = parameters.pop("totalSupply")
    
    tool_def = TOOL_DEFINITIONS[tool_name]
    endpoint = tool_def["endpoint"]
    method = tool_def["method"]
    
    # Handle URL parameters for GET requests
    if "{address}" in endpoint:
        if "address" in parameters:
            endpoint = endpoint.replace("{address}", parameters["address"])
            # For GET requests, remove address from parameters
            if method == "GET":
                parameters = {}
    
    # Prepare headers - check if Bearer token is needed
    headers = {}
    # Note: Add backend auth headers here if your Etherlink API requires them
    
    try:
        if method == "POST":
            response = requests.post(endpoint, json=parameters, headers=headers, timeout=60)
        elif method == "GET":
            response = requests.get(endpoint, headers=headers, timeout=60)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return {
            "success": True,
            "tool": tool_name,
            "result": response.json()
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "tool": tool_name,
            "error": str(e)
        }

def get_llm_tool_definitions(tool_names: List[str]) -> List[Dict[str, Any]]:
    """Convert tool definitions to OpenAI-compatible function-calling format (used by Groq)."""
    
    tools = []
    for tool_name in tool_names:
        if tool_name in TOOL_DEFINITIONS:
            tool_def = TOOL_DEFINITIONS[tool_name]
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_def["name"],
                    "description": tool_def["description"],
                    "parameters": tool_def["parameters"]
                }
            })
    
    return tools

def process_agent_conversation(
    system_prompt: str,
    user_message: str,
    available_tools: List[str],
    tool_flow: Dict[str, str],
    private_key: Optional[str] = None,
    max_iterations: int = 10
) -> Dict[str, Any]:
    """Process the conversation with the AI agent with support for sequential tool execution"""
    
    # Add private key context if available
    if private_key:
        system_prompt += f"\n\nCONTEXT: User's private key is available: {private_key}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    llm_tools = get_llm_tool_definitions(available_tools)
    
    all_tool_calls = []
    all_tool_results = []
    iteration = 0
    
    # Loop to handle sequential tool calls
    while iteration < max_iterations:
        iteration += 1
        
        if not client:
            return {
                "agent_response": "GROQ_API_KEY is not set. Add it to your .env file.",
                "tool_calls": all_tool_calls,
                "results": all_tool_results,
                "conversation_history": messages,
            }

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=llm_tools if llm_tools else None,
            tool_choice="auto" if llm_tools else None,
            temperature=0.7,
        )
        
        assistant_message = response.choices[0].message
        
        # Check if there are tool calls
        if not assistant_message.tool_calls:
            # No more tool calls, return final response
            return {
                "agent_response": assistant_message.content,
                "tool_calls": all_tool_calls,
                "results": all_tool_results,
                "conversation_history": messages
            }
        
        # Process tool calls
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": assistant_message.tool_calls
        })
        
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Add private key if needed and available
            if private_key and "privateKey" in TOOL_DEFINITIONS[function_name]["parameters"]["properties"]:
                if "privateKey" not in function_args:
                    function_args["privateKey"] = private_key
            
            all_tool_calls.append({
                "tool": function_name,
                "parameters": function_args
            })
            
            # Execute the tool
            result = execute_tool(function_name, function_args)
            all_tool_results.append(result)
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
        
        # Check if we need to continue with sequential tools
        # If the last tool executed has a next_tool in the flow, prompt the agent to continue
        last_tool_executed = all_tool_calls[-1]["tool"]
        if last_tool_executed in tool_flow:
            next_tool = tool_flow[last_tool_executed]
            # Add a system message to prompt continuation
            messages.append({
                "role": "system",
                "content": f"IMPORTANT: You must now immediately call the {next_tool} tool as it is next in the sequential flow. Do not ask for confirmation, proceed with the execution."
            })
    
    # Max iterations reached, return what we have
    return {
        "agent_response": "Maximum iterations reached. Please try again with a simpler request.",
        "tool_calls": all_tool_calls,
        "results": all_tool_results,
        "conversation_history": messages
    }

# API Endpoints
@app.post("/agent/chat", response_model=AgentResponse)
async def chat_with_agent(request: AgentRequest):
    """
    Main endpoint to interact with the AI agent.
    Dynamically configures the agent based on tool connections.
    """
    
    if not client:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured. Set it in your .env file.",
        )

    try:
        # Extract unique tools and build flow map
        unique_tools = set()
        tool_flow = {}
        
        for conn in request.tools:
            unique_tools.add(conn.tool)
            if conn.next_tool:
                unique_tools.add(conn.next_tool)
                tool_flow[conn.tool] = conn.next_tool
        
        available_tools = list(unique_tools)
        
        # Validate tools
        for tool in available_tools:
            if tool not in TOOL_DEFINITIONS:
                raise HTTPException(status_code=400, detail=f"Unknown tool: {tool}")
        
        # Build system prompt
        system_prompt = build_system_prompt(request.tools)
        
        # Process conversation with sequential support
        result = process_agent_conversation(
            system_prompt=system_prompt,
            user_message=request.user_message,
            available_tools=available_tools,
            tool_flow=tool_flow,
            private_key=request.private_key
        )
        
        return AgentResponse(
            agent_response=result["agent_response"],
            tool_calls=result["tool_calls"],
            results=result["results"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Etherlink AI Agent Builder"}

@app.get("/tools")
async def list_tools():
    """List all available tools"""
    return {
        "tools": list(TOOL_DEFINITIONS.keys()),
        "details": TOOL_DEFINITIONS
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
