# This file defines the Google Cloud resources for your scheduled job.

# Configure the Google Cloud provider
provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Enable the necessary APIs for the project.
# Terraform needs these APIs to be active to create the resources.
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudscheduler.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com" # Needed for Cloud Run to build images
  ])
  project                    = var.project_id
  service                    = each.key
  disable_on_destroy         = false
}

# 2. Create an Artifact Registry repository to store your Docker images.
resource "google_artifact_registry_repository" "fpl_2025" {
  project       = var.project_id
  location      = var.region
  repository_id = var.repository_id
  description   = "Docker repository for FPL 2025"
  format        = "DOCKER"
  depends_on = [
    google_project_service.apis
  ]
}

# 3. Create a Service Account for the Cloud Scheduler job to use.
# This gives the scheduler a specific identity with limited permissions.
# resource "google_service_account" "scheduler_invoker" {
#   project      = var.project_id
#   account_id   = "scheduler-job-invoker"
#   display_name = "Service Account for Cloud Scheduler to invoke Cloud Run"
# }
#
# # 4. Create the Cloud Run Job.
# # This resource defines the containerized task that will be executed.
# # IMPORTANT: You must update the 'image' attribute after you build and push your Docker image.
# resource "google_cloud_run_v2_job" "python_job" {
#   project  = var.project_id
#   location = var.region
#   name     = "scheduled-python-job"
#
#   template {
#     template {
#       containers {
#         # --- ACTION REQUIRED ---
#         # Replace this with the full URL of your image in Artifact Registry
#         # Example: "us-central1-docker.pkg.dev/your-gcp-project-id/my-docker-repo/my-python-app:latest"
#         image = "gcr.io/cloudrun/hello" # Placeholder image
#       }
#     }
#   }
#   depends_on = [
#     google_project_service.apis
#   ]
# }
#
# # 5. Grant the Scheduler's Service Account permission to run the Cloud Run Job.
# resource "google_cloud_run_v2_job_iam_member" "job_invoker_permission" {
#   project  = var.project_id
#   location = var.region
#   name     = google_cloud_run_v2_job.python_job.name
#   role     = "roles/run.invoker"
#   member   = "serviceAccount:${google_service_account.scheduler_invoker.email}"
# }
#
# # 6. Create the Cloud Scheduler job to trigger the Cloud Run Job.
# resource "google_cloud_scheduler_job" "schedule" {
#   project   = var.project_id
#   region    = var.region
#   name      = "trigger-python-job-daily"
#   schedule  = "0 5 * * *" # Runs every day at 5:00 AM (in UTC)
#   time_zone = "Etc/UTC"
#
#   http_target {
#     uri = "https://run.googleapis.com/v2/${google_cloud_run_v2_job.python_job.id}:run"
#
#     oauth_token {
#       service_account_email = google_service_account.scheduler_invoker.email
#     }
#   }
#
#   depends_on = [
#     google_project_service.apis,
#     google_cloud_run_v2_job_iam_member.job_invoker_permission
#   ]
# }
