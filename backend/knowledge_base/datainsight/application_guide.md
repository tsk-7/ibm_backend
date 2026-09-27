# DataInsight Studio Application Guide

DataInsight Studio provides dataset upload, listing, preview, profiling, preprocessing, statistics, outlier reports, visualization recommendations, health scoring, history, and download through the existing FastAPI backend. The React frontend should use the dataset ID returned by upload as the active dataset ID.

The AI assistant is an orchestration layer over existing backend services. Dataset facts come from current service results; the local knowledge base provides general analytics guidance. Destructive actions require a separate user confirmation and are verified after execution.