# AI Chatbot Application

A full-stack AI chatbot application with React frontend, FastAPI backend, and PyTorch-based language model integration.

## Architecture

- **Frontend**: React 18 with TypeScript
- **Backend**: FastAPI with Python 3.11
- **AI Engine**: PyTorch with Transformers
- **Database**: SQLite (development) / PostgreSQL (production)
- **Containerization**: Docker Compose

## Project Structure

```
chatbotR1/
├── frontend/              # React application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API services
│   │   ├── types/         # TypeScript types
│   │   └── utils/         # Utility functions
│   ├── public/            # Static assets
│   ├── package.json       # Node.js dependencies
│   └── Dockerfile         # Frontend container config
├── backend/               # FastAPI application
│   ├── app/
│   │   ├── routers/       # API route handlers
│   │   ├── models/        # Database models
│   │   ├── services/      # Business logic
│   │   └── database/      # Database configuration
│   ├── tests/             # Backend tests
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile         # Backend container config
├── shared/                # Shared resources
│   ├── models/            # AI model files
│   └── data/              # Database and data files
├── docker-compose.yml     # Container orchestration
├── .env.example           # Environment variables template
├── .env.development       # Development environment
└── README.md              # This file
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Git

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd courses/chatbotR1
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration if needed
   ```

3. **Start the application:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Development Setup

For development with hot reloading:

```bash
# Start in development mode
docker-compose -f docker-compose.yml up --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Environment Variables

### Frontend Variables
- `REACT_APP_API_URL`: Backend API URL (default: http://localhost:8000)
- `REACT_APP_WS_URL`: WebSocket URL for real-time features

### Backend Variables
- `DATABASE_URL`: Database connection string
- `MODEL_PATH`: Path to AI model files
- `MAX_CONTEXT_LENGTH`: Maximum conversation context length
- `DEVICE`: PyTorch device (cpu/cuda)
- `DEBUG`: Enable debug mode
- `LOG_LEVEL`: Logging level (debug/info/warning/error)

## API Endpoints

### Chat Endpoints
- `POST /api/chat/send` - Send message and get AI response
- `GET /api/chat/conversations` - List all conversations
- `GET /api/chat/conversations/{id}` - Get conversation messages
- `POST /api/chat/conversations` - Create new conversation
- `DELETE /api/chat/conversations/{id}` - Delete conversation

### Health Endpoints
- `GET /api/health` - Application health check
- `GET /api/model/info` - AI model information

## Development

### Frontend Development
```bash
cd frontend
npm install
npm start
```

### Backend Development
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Running Tests
```bash
# Frontend tests
cd frontend && npm test

# Backend tests
cd backend && pytest
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 3000 and 8000 are available
2. **Model download**: First run may take time to download AI models
3. **Memory issues**: Ensure sufficient RAM for PyTorch models
4. **Docker issues**: Try `docker-compose down -v` to reset volumes

### Logs
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs frontend
docker-compose logs backend
```

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation as needed
4. Use TypeScript for frontend code
5. Follow Python type hints for backend code

## License

This project is for educational purposes.