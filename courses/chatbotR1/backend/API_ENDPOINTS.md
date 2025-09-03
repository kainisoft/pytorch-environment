# Chat API Endpoints Documentation

This document describes the implemented chat API endpoints for the AI Chatbot application.

## Base URL
All endpoints are prefixed with `/api/chat`

## Endpoints

### 1. Send Message
**POST** `/api/chat/send`

Send a message to the AI chatbot and receive a response.

#### Request Body
```json
{
  "message": "Hello, how are you?",
  "conversation_id": "optional-conversation-id"
}
```

#### Response
```json
{
  "response": "AI generated response",
  "conversation_id": "conversation-uuid",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Behavior
- If `conversation_id` is provided, validates the conversation exists
- If `conversation_id` is not provided, creates a new conversation
- Stores both user message and AI response in the database
- Auto-generates conversation title from first message

---

### 2. Get Conversations List
**GET** `/api/chat/conversations`

Retrieve a list of all conversations with metadata.

#### Query Parameters
- `limit` (optional): Maximum number of conversations to return (default: 50)
- `offset` (optional): Number of conversations to skip (default: 0)

#### Response
```json
{
  "conversations": [
    {
      "id": "conversation-uuid",
      "title": "Conversation Title",
      "updated_at": "2024-01-01T12:00:00Z",
      "message_count": 5
    }
  ]
}
```

#### Behavior
- Returns conversations ordered by `updated_at` (most recent first)
- Includes message count for each conversation
- Supports pagination via `limit` and `offset`

---

### 3. Get Conversation Messages
**GET** `/api/chat/conversations/{conversation_id}`

Retrieve all messages for a specific conversation.

#### Path Parameters
- `conversation_id`: The UUID of the conversation

#### Query Parameters
- `limit` (optional): Maximum number of messages to return (default: 100)
- `offset` (optional): Number of messages to skip (default: 0)

#### Response
```json
{
  "conversation_id": "conversation-uuid",
  "title": "Conversation Title",
  "messages": [
    {
      "id": 1,
      "content": "Hello, how are you?",
      "is_user": true,
      "timestamp": "2024-01-01T12:00:00Z"
    },
    {
      "id": 2,
      "content": "I'm doing well, thank you!",
      "is_user": false,
      "timestamp": "2024-01-01T12:00:05Z"
    }
  ]
}
```

#### Behavior
- Returns 404 if conversation doesn't exist
- Messages are ordered chronologically (oldest first)
- Supports pagination for large conversations

---

### 4. Create New Conversation
**POST** `/api/chat/conversations`

Create a new empty conversation.

#### Request Body
```json
{
  "title": "My New Conversation"
}
```

#### Response
```json
{
  "id": "new-conversation-uuid",
  "title": "My New Conversation",
  "created_at": "2024-01-01T12:00:00Z"
}
```

#### Behavior
- Creates an empty conversation with the specified title
- Returns the new conversation details

---

### 5. Delete Conversation
**DELETE** `/api/chat/conversations/{conversation_id}`

Delete a conversation and all its messages.

#### Path Parameters
- `conversation_id`: The UUID of the conversation to delete

#### Response
```json
{
  "success": true,
  "message": "Conversation {conversation_id} deleted successfully"
}
```

#### Behavior
- Returns 404 if conversation doesn't exist
- Deletes all messages in the conversation (cascade delete)
- Returns success confirmation

---

## Error Responses

All endpoints return standardized error responses:

```json
{
  "detail": "Error description"
}
```

### Common HTTP Status Codes
- `200`: Success
- `404`: Resource not found (conversation doesn't exist)
- `422`: Validation error (invalid request data)
- `500`: Internal server error

---

## Request/Response Models

### Request Models

#### ChatMessageRequest
- `message` (string, required): User message content (1-4000 characters)
- `conversation_id` (string, optional): Existing conversation ID

#### ConversationCreateRequest
- `title` (string, required): Conversation title (1-255 characters)

### Response Models

#### MessageResponse
- `id` (integer): Message ID
- `content` (string): Message content
- `is_user` (boolean): True if user message, false if AI message
- `timestamp` (datetime): Message timestamp

#### ChatResponse
- `response` (string): AI generated response
- `conversation_id` (string): Conversation UUID
- `timestamp` (datetime): Response timestamp

#### ConversationSummary
- `id` (string): Conversation UUID
- `title` (string): Conversation title
- `updated_at` (datetime): Last update timestamp
- `message_count` (integer): Number of messages in conversation

#### ConversationCreateResponse
- `id` (string): New conversation UUID
- `title` (string): Conversation title
- `created_at` (datetime): Creation timestamp

#### DeleteResponse
- `success` (boolean): Deletion success status
- `message` (string): Confirmation message

---

## Database Integration

The API endpoints integrate with the following database operations:

### Conversations Table
- Stores conversation metadata (ID, title, timestamps)
- Auto-generates UUIDs for conversation IDs
- Tracks creation and update timestamps

### Messages Table
- Stores individual messages with conversation references
- Tracks user vs AI messages with `is_user` flag
- Maintains chronological order with timestamps
- Cascade deletes when conversation is deleted

### CRUD Operations
- Uses `ConversationCRUD` for conversation operations
- Uses `MessageCRUD` for message operations
- Implements proper error handling and logging
- Supports pagination for large datasets

---

## Future Enhancements

The current implementation provides a solid foundation and can be extended with:

1. **AI Integration**: Replace placeholder responses with actual PyTorch model inference
2. **Authentication**: Add user authentication and conversation ownership
3. **Real-time Updates**: Implement WebSocket support for live chat
4. **Message Search**: Add full-text search across conversations
5. **Export/Import**: Add conversation export and import functionality
6. **Rate Limiting**: Implement request rate limiting per user
7. **Message Reactions**: Add support for message reactions and feedback
8. **Conversation Sharing**: Enable conversation sharing between users

---

## Testing

The implementation includes validation scripts:

- `validate_structure.py`: Validates Python syntax and required elements
- `test_chat_endpoints.py`: Comprehensive endpoint testing (requires dependencies)

To run validation:
```bash
python3 validate_structure.py
```

The API endpoints are ready for integration testing once the FastAPI dependencies are installed and the database is initialized.