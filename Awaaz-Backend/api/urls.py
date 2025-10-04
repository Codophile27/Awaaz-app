from django.urls import path
from .views import AnalyzeView, ComplaintCreateView, UploadComplaintView, PostComplaintView


urlpatterns = [
    path('analyze/', AnalyzeView.as_view(), name='api_analyze'),
    path('complaints/', ComplaintCreateView.as_view(), name='api_complaint_create'),
    path('upload_complaint/', UploadComplaintView.as_view(), name='api_upload_complaint'),
    path('post_complaint/', PostComplaintView.as_view(), name='api_post_complaint'),
]

