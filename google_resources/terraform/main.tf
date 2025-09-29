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
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# IAM Service module to manage GitHub Actions authentication
module "iam_service" {
  source      = "./iam_service"
  project_id  = var.project_id
  github_repo = var.github_repo
}

# Artifact Registry module to manage Docker repository and GitHub Actions integration
module "artifact_registry" {
  source                               = "./artifact_registry"
  project_id                           = var.project_id
  region                               = var.region
  repository_id                        = var.repository_id
  api_services_dependency              = google_project_service.apis
  github_actions_service_account_email = module.iam_service.github_actions_service_account_email
}

# Cloud Run module to manage the containerized job
module "cloud_run" {
  source                  = "./cloud_run"
  project_id              = var.project_id
  region                  = var.region
  # container_image =       "us-docker.pkg.dev/cloudrun/container/hello"
  container_image         = "${module.artifact_registry.repository_url}/fpl-app:latest"
  api_services_dependency = google_project_service.apis
}

# Cloud Scheduler module to manage scheduled job execution
module "cloud_scheduler" {
  source             = "./cloud_scheduler"
  project_id         = var.project_id
  region             = var.region
  cloud_run_job_name = module.cloud_run.job_name
  cloud_run_job_id   = module.cloud_run.job_id

  depends_on = [
    google_project_service.apis,
    module.cloud_run
  ]
  scheduler_service_account = ""
}
