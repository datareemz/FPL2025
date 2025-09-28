# Cloud Scheduler resources

resource "google_service_account" "scheduler_invoker" {
  project      = var.project_id
  account_id   = "scheduler-job-invoker"
  display_name = "Service Account for Cloud Scheduler to invoke Cloud Run"
}

# Create the Cloud Scheduler job to trigger the Cloud Run Job.
resource "google_cloud_scheduler_job" "schedule" {
  project   = var.project_id
  region    = var.region
  name      = "fpl-2025-scheduler-job"
  schedule  = var.cron_schedule
  time_zone = "Etc/UTC"

  http_target {
    uri = "https://run.googleapis.com/v2/${var.cloud_run_job_id}:run"

    oauth_token {
      service_account_email = var.scheduler_service_account
    }
  }
}