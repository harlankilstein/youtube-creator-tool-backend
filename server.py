from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import openai
import requests
import re
import json
from googleapiclient.discovery import build
from textstat import flesch_reading_ease, flesch_kincaid_grade
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# API Configuration
openai.api_key = os.environ['OPENAI_API_KEY']
YOUTUBE_API_KEY = os.environ['YOUTUBE_API_KEY']
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="CreatorFlow API", description="YouTube Creator Optimization Platform")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Pydantic Models
class HookGeneratorRequest(BaseModel):
    topic: str
    target_audience: Optional[str] = "General audience"
    content_type: Optional[str] = "Educational"
    youtube_url: Optional[str] = None

class Hook(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    strength_score: int
    psychological_triggers: List[str]
    reasoning: str

class HookGeneratorResponse(BaseModel):
    hooks: List[Hook]
    analysis_summary: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class EmotionalAnalysisRequest(BaseModel):
    script_text: str
    youtube_url: Optional[str] = None

class EmotionalPoint(BaseModel):
    timestamp: int
    emotion: str
    intensity: float
    description: str

class EmotionalAnalysisResponse(BaseModel):
    overall_score: float
    emotional_journey: List[EmotionalPoint]
    dominant_emotions: List[str]
    improvement_suggestions: List[str]
    psychological_triggers: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SEOAnalysisRequest(BaseModel):
    title: str
    description: str
    tags: List[str]
    script_content: Optional[str] = ""
    youtube_url: Optional[str] = None

class SEOAnalysisResponse(BaseModel):
    overall_seo_score: float
    title_score: float
    description_score: float
    keyword_optimization: Dict[str, Any]
    competition_analysis: Dict[str, Any]
    trending_topics: List[str]
    recommendations: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ScriptGeneratorRequest(BaseModel):
    topic: str
    target_style: str
    competitor_channels: Optional[List[str]] = []
    target_length: Optional[int] = 1000
    youtube_reference: Optional[str] = None

class ScriptGeneratorResponse(BaseModel):
    original_script: str
    style_analysis: str
    seo_integration: List[str]
    uniqueness_score: float
    engagement_predictions: Dict[str, float]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TitleOptimizerRequest(BaseModel):
    content_summary: str
    target_keywords: List[str]
    audience_demographics: Optional[str] = "General"

class OptimizedTitle(BaseModel):
    title: str
    clickability_score: float
    seo_score: float
    emotional_triggers: List[str]
    predicted_ctr: float

class TitleOptimizerResponse(BaseModel):
    optimized_titles: List[OptimizedTitle]
    analysis_summary: str
    a_b_testing_recommendations: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RetentionPlanningRequest(BaseModel):
    script_content: str
    video_length_minutes: int
    target_audience: str

class RetentionPoint(BaseModel):
    timestamp: int
    predicted_retention: float
    risk_level: str
    recommendations: List[str]

class RetentionPlanningResponse(BaseModel):
    overall_retention_prediction: float
    critical_points: List[RetentionPoint]
    pacing_analysis: Dict[str, Any]
    engagement_strategies: List[str]
    drop_off_predictions: List[Dict[str, Any]]
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Helper Functions
def extract_video_id(youtube_url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None

async def get_video_details(video_id: str) -> Optional[Dict]:
    """Get video details from YouTube API"""
    try:
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        )
        response = request.execute()
        
        if response['items']:
            return response['items'][0]
        return None
    except Exception as e:
        logging.error(f"Error fetching video details: {e}")
        return None

async def analyze_with_gpt(prompt: str, system_message: str = "You are an expert YouTube content analyst.") -> str:
    """Analyze content using GPT-4"""
    try:
        response = await asyncio.to_thread(
            openai.chat.completions.create,
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error with GPT analysis: {e}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

# API Endpoints

@api_router.post("/hook-generator", response_model=HookGeneratorResponse)
async def generate_hooks(request: HookGeneratorRequest):
    """Generate compelling opening hooks with psychological triggers"""
    
    # Get video context if URL provided
    video_context = ""
    if request.youtube_url:
        video_id = extract_video_id(request.youtube_url)
        if video_id:
            video_details = await get_video_details(video_id)
            if video_details:
                video_context = f"Reference video: {video_details['snippet']['title']} - {video_details['snippet']['description'][:200]}"
    
    prompt = f"""
    Create 10 compelling opening hooks for a YouTube video with these specifications:
    
    Topic: {request.topic}
    Target Audience: {request.target_audience}
    Content Type: {request.content_type}
    {video_context}
    
    For each hook, provide:
    1. The hook text (1-2 sentences)
    2. A strength score (1-100)
    3. Psychological triggers used (curiosity gap, urgency, social proof, etc.)
    4. Brief reasoning for the score
    
    Focus on psychological triggers like:
    - Curiosity gaps
    - Pattern interrupts
    - Social proof
    - Urgency/scarcity
    - Controversy/contrarian views
    - Personal storytelling
    - Problem/solution frameworks
    
    Format as JSON with this structure:
    {{
        "hooks": [
            {{
                "text": "hook text here",
                "strength_score": 85,
                "psychological_triggers": ["curiosity_gap", "urgency"],
                "reasoning": "explanation here"
            }}
        ],
        "analysis_summary": "Overall analysis of hook strategy"
    }}
    """
    
    response = await analyze_with_gpt(prompt, "You are an expert YouTube hook writer who understands viewer psychology and engagement patterns.")
    
    try:
        result = json.loads(response)
        hooks = [
            Hook(
                text=hook["text"],
                strength_score=hook["strength_score"],
                psychological_triggers=hook["psychological_triggers"],
                reasoning=hook["reasoning"]
            )
            for hook in result["hooks"]
        ]
        
        # Save to database
        hook_analysis = HookGeneratorResponse(
            hooks=hooks,
            analysis_summary=result["analysis_summary"]
        )
        
        await db.hook_analyses.insert_one(hook_analysis.dict())
        
        return hook_analysis
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")

@api_router.post("/emotional-analysis", response_model=EmotionalAnalysisResponse)
async def analyze_emotional_triggers(request: EmotionalAnalysisRequest):
    """Analyze emotional impact and journey of content"""
    
    prompt = f"""
    Analyze the emotional journey and psychological impact of this script:
    
    Script: {request.script_text}
    
    Provide a comprehensive emotional analysis including:
    
    1. Overall emotional score (0-100)
    2. Emotional journey mapping (divide script into segments and identify emotions)
    3. Dominant emotions present
    4. Improvement suggestions for emotional impact
    5. Psychological triggers detected
    
    For the emotional journey, estimate timestamps and provide:
    - Emotion type (excitement, curiosity, tension, satisfaction, etc.)
    - Intensity (0.0-1.0)
    - Description of what creates this emotion
    
    Format as JSON:
    {{
        "overall_score": 75.5,
        "emotional_journey": [
            {{
                "timestamp": 0,
                "emotion": "curiosity",
                "intensity": 0.8,
                "description": "Opening hook creates strong curiosity"
            }}
        ],
        "dominant_emotions": ["curiosity", "excitement"],
        "improvement_suggestions": ["Add more tension", "Include personal story"],
        "psychological_triggers": ["curiosity_gap", "social_proof"]
    }}
    """
    
    response = await analyze_with_gpt(prompt, "You are an expert in emotional psychology and content engagement analysis.")
    
    try:
        result = json.loads(response)
        
        emotional_points = [
            EmotionalPoint(
                timestamp=point["timestamp"],
                emotion=point["emotion"],
                intensity=point["intensity"],
                description=point["description"]
            )
            for point in result["emotional_journey"]
        ]
        
        analysis = EmotionalAnalysisResponse(
            overall_score=result["overall_score"],
            emotional_journey=emotional_points,
            dominant_emotions=result["dominant_emotions"],
            improvement_suggestions=result["improvement_suggestions"],
            psychological_triggers=result["psychological_triggers"]
        )
        
        await db.emotional_analyses.insert_one(analysis.dict())
        return analysis
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")

@api_router.post("/seo-analysis", response_model=SEOAnalysisResponse)
async def analyze_seo(request: SEOAnalysisRequest):
    """Comprehensive SEO analysis and optimization recommendations"""
    
    prompt = f"""
    Perform comprehensive SEO analysis for this YouTube content:
    
    Title: {request.title}
    Description: {request.description}
    Tags: {', '.join(request.tags)}
    Script Content: {request.script_content[:1000]}...
    
    Analyze and score:
    1. Overall SEO score (0-100)
    2. Title optimization score (0-100)
    3. Description optimization score (0-100)
    4. Keyword optimization analysis
    5. Competition analysis insights
    6. Trending topics integration
    7. Specific recommendations for improvement
    
    Consider:
    - Keyword density and placement
    - Search intent alignment
    - Click-through rate optimization
    - Trending topics and seasonal patterns
    - Competition analysis
    - Long-tail keyword opportunities
    
    Format as JSON:
    {{
        "overall_seo_score": 72.5,
        "title_score": 85.0,
        "description_score": 68.0,
        "keyword_optimization": {{
            "primary_keywords": ["keyword1", "keyword2"],
            "keyword_density": 2.5,
            "placement_score": 80
        }},
        "competition_analysis": {{
            "competition_level": "medium",
            "ranking_difficulty": 65,
            "opportunities": ["long-tail keywords", "seasonal content"]
        }},
        "trending_topics": ["topic1", "topic2"],
        "recommendations": ["recommendation1", "recommendation2"]
    }}
    """
    
    response = await analyze_with_gpt(prompt, "You are an expert YouTube SEO specialist with deep knowledge of search algorithms and ranking factors.")
    
    try:
        result = json.loads(response)
        
        analysis = SEOAnalysisResponse(
            overall_seo_score=result["overall_seo_score"],
            title_score=result["title_score"],
            description_score=result["description_score"],
            keyword_optimization=result["keyword_optimization"],
            competition_analysis=result["competition_analysis"],
            trending_topics=result["trending_topics"],
            recommendations=result["recommendations"]
        )
        
        await db.seo_analyses.insert_one(analysis.dict())
        return analysis
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")

@api_router.post("/script-generator", response_model=ScriptGeneratorResponse)
async def generate_original_script(request: ScriptGeneratorRequest):
    """Generate original scripts based on successful patterns"""
    
    # Analyze competitor channels if provided
    competitor_context = ""
    if request.competitor_channels:
        competitor_context = f"Analyze and emulate successful patterns from these channels: {', '.join(request.competitor_channels)}"
    
    prompt = f"""
    Generate an original, engaging YouTube script with these specifications:
    
    Topic: {request.topic}
    Target Style: {request.target_style}
    Target Length: {request.target_length} words
    {competitor_context}
    
    Requirements:
    1. Create completely original content (no plagiarism)
    2. Incorporate proven engagement patterns
    3. Include SEO-optimized elements
    4. Structure for maximum retention
    5. Use psychological triggers effectively
    
    The script should include:
    - Compelling hook (first 15 seconds)
    - Clear value proposition
    - Structured main content with engagement points
    - Strong call-to-action
    - Natural keyword integration
    
    Provide:
    1. The complete original script
    2. Style analysis explanation
    3. SEO integration points
    4. Uniqueness verification
    5. Engagement predictions (hook strength, retention prediction, etc.)
    
    Format as JSON:
    {{
        "original_script": "full script here...",
        "style_analysis": "explanation of style choices",
        "seo_integration": ["keyword1 placement", "keyword2 strategy"],
        "uniqueness_score": 95.5,
        "engagement_predictions": {{
            "hook_strength": 85,
            "retention_prediction": 78,
            "ctr_prediction": 12.5
        }}
    }}
    """
    
    response = await analyze_with_gpt(prompt, "You are an expert YouTube scriptwriter with deep knowledge of successful content patterns and viewer psychology.")
    
    try:
        result = json.loads(response)
        
        analysis = ScriptGeneratorResponse(
            original_script=result["original_script"],
            style_analysis=result["style_analysis"],
            seo_integration=result["seo_integration"],
            uniqueness_score=result["uniqueness_score"],
            engagement_predictions=result["engagement_predictions"]
        )
        
        await db.script_generations.insert_one(analysis.dict())
        return analysis
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")

@api_router.post("/title-optimizer", response_model=TitleOptimizerResponse)
async def optimize_titles(request: TitleOptimizerRequest):
    """Generate and optimize multiple title variations"""
    
    prompt = f"""
    Create 10+ optimized YouTube title variations based on:
    
    Content Summary: {request.content_summary}
    Target Keywords: {', '.join(request.target_keywords)}
    Audience Demographics: {request.audience_demographics}
    
    For each title, analyze and score:
    1. Clickability score (0-100) - psychological appeal
    2. SEO score (0-100) - search optimization
    3. Emotional triggers used
    4. Predicted click-through rate (%)
    
    Consider:
    - Emotional triggers (curiosity, urgency, benefit-driven)
    - Keyword placement and density
    - Character length optimization
    - Power words and phrases
    - A/B testing potential
    - Trending patterns
    
    Provide A/B testing recommendations for the top titles.
    
    Format as JSON:
    {{
        "optimized_titles": [
            {{
                "title": "Title text here",
                "clickability_score": 85.5,
                "seo_score": 78.0,
                "emotional_triggers": ["curiosity", "urgency"],
                "predicted_ctr": 12.8
            }}
        ],
        "analysis_summary": "Overall title strategy explanation",
        "a_b_testing_recommendations": ["Test emotional vs rational", "Compare question vs statement format"]
    }}
    """
    
    response = await analyze_with_gpt(prompt, "You are an expert YouTube title optimizer with deep knowledge of psychology, SEO, and click-through rate optimization.")
    
    try:
        result = json.loads(response)
        
        titles = [
            OptimizedTitle(
                title=title["title"],
                clickability_score=title["clickability_score"],
                seo_score=title["seo_score"],
                emotional_triggers=title["emotional_triggers"],
                predicted_ctr=title["predicted_ctr"]
            )
            for title in result["optimized_titles"]
        ]
        
        analysis = TitleOptimizerResponse(
            optimized_titles=titles,
            analysis_summary=result["analysis_summary"],
            a_b_testing_recommendations=result["a_b_testing_recommendations"]
        )
        
        await db.title_optimizations.insert_one(analysis.dict())
        return analysis
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")

@api_router.post("/retention-planning", response_model=RetentionPlanningResponse)
async def plan_retention_strategy(request: RetentionPlanningRequest):
    """Analyze and optimize content for viewer retention"""
    
    # Calculate readability scores
    readability_score = flesch_reading_ease(request.script_content)
    grade_level = flesch_kincaid_grade(request.script_content)
    
    prompt = f"""
    Perform comprehensive retention analysis for this YouTube content:
    
    Script Content: {request.script_content}
    Video Length: {request.video_length_minutes} minutes
    Target Audience: {request.target_audience}
    Readability Score: {readability_score}
    Grade Level: {grade_level}
    
    Analyze and provide:
    1. Overall retention prediction (0-100%)
    2. Critical drop-off points with timestamps
    3. Risk levels for each section (high/medium/low)
    4. Specific recommendations for each critical point
    5. Pacing analysis (too fast/slow sections)
    6. Engagement strategies placement
    7. Drop-off predictions with reasoning
    
    Consider:
    - Content structure and flow
    - Information density
    - Engagement techniques placement
    - Audience attention patterns
    - Psychological engagement principles
    - Visual/audio cues timing
    
    Format as JSON:
    {{
        "overall_retention_prediction": 72.5,
        "critical_points": [
            {{
                "timestamp": 45,
                "predicted_retention": 68.5,
                "risk_level": "medium",
                "recommendations": ["Add visual element", "Increase pacing"]
            }}
        ],
        "pacing_analysis": {{
            "overall_pace": "optimal",
            "slow_sections": [{"start": 120, "end": 180, "recommendation": "Add engagement hook"}],
            "fast_sections": []
        }},
        "engagement_strategies": ["Use pattern interrupt at 2:30", "Add story element at 4:15"],
        "drop_off_predictions": [
            {{
                "timestamp": 90,
                "predicted_drop": 15.2,
                "reason": "Information overload",
                "fix": "Break into smaller chunks"
            }}
        ]
    }}
    """
    
    response = await analyze_with_gpt(prompt, "You are an expert YouTube retention specialist with deep knowledge of viewer behavior patterns and engagement psychology.")
    
    try:
        result = json.loads(response)
        
        critical_points = [
            RetentionPoint(
                timestamp=point["timestamp"],
                predicted_retention=point["predicted_retention"],
                risk_level=point["risk_level"],
                recommendations=point["recommendations"]
            )
            for point in result["critical_points"]
        ]
        
        analysis = RetentionPlanningResponse(
            overall_retention_prediction=result["overall_retention_prediction"],
            critical_points=critical_points,
            pacing_analysis=result["pacing_analysis"],
            engagement_strategies=result["engagement_strategies"],
            drop_off_predictions=result["drop_off_predictions"]
        )
        
        await db.retention_analyses.insert_one(analysis.dict())
        return analysis
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")

# Basic status endpoints
@api_router.get("/")
async def root():
    return {"message": "CreatorFlow API is running", "version": "1.0.0"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
