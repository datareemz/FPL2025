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
resource "google_artifact_registry_repository" "fpl_2025_repo" {
  project       = var.project_id
  location      = var.region
  repository_id = var.repository_id
  description   = "Docker repository for fpl_2025 jobs"
  format        = "DOCKER"
  depends_on = [
    google_project_service.apis
  ]
}

# 3a. Create a Service Account for the GitHub Actions runner.
# This identity will be assumed by GitHub to push to Artifact Registry.
resource "google_service_account" "github_actions_writer" {
  project      = var.project_id
  account_id   = "github-actions-writer"
  display_name = "Service Account for GitHub Actions"
  description  = "Allows GitHub Actions to push images to Artifact Registry"
}

# 3b. Create the Workload Identity Pool.
# This acts as a container for external identity providers.
resource "google_iam_workload_identity_pool" "github_pool" {
  project                   = var.project_id
  workload_identity_pool_id = "github-actions-pool"
  display_name              = "GitHub Actions Pool"
  description               = "Trusts identities from GitHub Actions"
  depends_on = [
    google_project_service.apis
  ]
}

# 3c. Create the Workload Identity Provider.
# This configures the trust relationship specifically with GitHub.
resource "google_iam_workload_identity_pool_provider" "github_provider" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  display_name                       = "GitHub OIDC Provider"
  description                        = "Trusts GitHub based on OIDC tokens"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
  }

  attribute_condition = "attribute.repository == \"${var.github_repo}\""
  depends_on = [
    google_iam_workload_identity_pool.github_pool
  ]
}

# 3d. Grant the GitHub Actions Service Account permission to be impersonated by GitHub.
# This links your GitHub repo identity to the GCP Service Account.
resource "google_service_account_iam_member" "github_workload_user" {
  service_account_id = google_service_account.github_actions_writer.name
  role               = "roles/iam.workloadIdentityUser"
  # This principalSet identifies runners in your specific GitHub repo.
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_pool.name}/attribute.repository/${var.github_repo}"
}

# 3e. Grant the GitHub Actions Service Account permission to write to Artifact Registry.
resource "google_artifact_registry_repository_iam_member" "writer" {
  project    = var.project_id
  location   = google_artifact_registry_repository.fpl_2025_repo.location
  repository = google_artifact_registry_repository.fpl_2025_repo.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.github_actions_writer.email}"
}


# 3. Create a Service Account for the Cloud Scheduler job to use.
# This gives the scheduler a specific identity with limited permissions.
resource "google_service_account" "scheduler_invoker" {
  project      = var.project_id
  account_id   = "scheduler-job-invoker"
  display_name = "Service Account for Cloud Scheduler to invoke Cloud Run"
}

# 4. Create the Cloud Run Job.
# This resource defines the containerized task that will be executed.
# IMPORTANT: You must update the 'image' attribute after you build and push your Docker image.
resource "google_cloud_run_v2_job" "python_job" {
  project  = var.project_id
  location = var.region
  name     = "scheduled-python-job"

  template {
    template {
      containers {
        # --- ACTION REQUIRED ---
        # Replace this with the full URL of your image in Artifact Registry
        # Example: "us-central1-docker.pkg.dev/your-gcp-project-id/my-docker-repo/my-python-app:latest"
        image = "us-central1-docker.pkg.dev/ardent-quarter-468720-i2/fpl-2025/fpl-app:latest" # Placeholder image
      }
    }
  }
  depends_on = [
    google_project_service.apis
  ]
}

# 5. Grant the Scheduler's Service Account permission to run the Cloud Run Job.
resource "google_cloud_run_v2_job_iam_member" "job_invoker_permission" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_job.python_job.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}

# 6. Create the Cloud Scheduler job to trigger the Cloud Run Job.
resource "google_cloud_scheduler_job" "schedule" {
  project   = var.project_id
  region    = var.region
  name      = "trigger-python-job-daily"
  schedule  = "0 5 * * *" # Runs every day at 5:00 AM (in UTC)
  time_zone = "Etc/UTC"

  http_target {
    uri = "https://run.googleapis.com/v2/${google_cloud_run_v2_job.python_job.id}:run"

    oauth_token {
      service_account_email = google_service_account.scheduler_invoker.email
    }
  }

  depends_on = [
    google_project_service.apis,
    google_cloud_run_v2_job_iam_member.job_invoker_permission
  ]
}

