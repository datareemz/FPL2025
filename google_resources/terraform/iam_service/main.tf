# IAM Service resources for GitHub Actions integration

# Create a Workload Identity Pool for GitHub Actions
resource "google_iam_workload_identity_pool" "github_pool" {
  project                   = var.project_id
  workload_identity_pool_id = "github-actions-pool-new"
  display_name              = "GitHub Actions Pool New"
  description               = "Workload Identity Pool for GitHub Actions"
}

# Create a Workload Identity Provider for GitHub
resource "google_iam_workload_identity_pool_provider" "github_provider" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  display_name                       = "GitHub Provider"
  description                        = "OIDC identity provider for GitHub Actions"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
  }
  attribute_condition = "attribute.repository == \"${var.github_repo}\""
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Create a Service Account for GitHub Actions
resource "google_service_account" "github_actions_writer" {
  project      = var.project_id
  account_id   = "github-actions-writer"
  display_name = "Service Account for GitHub Actions"
  description  = "Service account that GitHub Actions will impersonate"
}

# Grant the GitHub Actions Service Account permission to be impersonated by GitHub
resource "google_service_account_iam_member" "github_workload_user" {
  service_account_id = google_service_account.github_actions_writer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_pool.name}/attribute.repository/${var.github_repo}"
}

# Create a Service Account for Cloud Run Job
resource "google_service_account" "cloud_run_job_sa" {
  project      = var.project_id
  account_id   = "cloud-run-fpl-job-sa"
  display_name = "Service Account for Cloud Run FPL Job"
  description  = "Service account used by Cloud Run job to access GCS and Secret Manager"
}

# Grant Cloud Run Service Account access to Cloud Storage bucket
resource "google_storage_bucket_iam_member" "cloud_run_bucket_access" {
  count  = var.bucket_name != "" ? 1 : 0
  bucket = var.bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cloud_run_job_sa.email}"
}

# Grant Cloud Run Service Account access to Secret Manager secret
resource "google_secret_manager_secret_iam_member" "cloud_run_secret_access" {
  count     = var.secret_name != "" ? 1 : 0
  project   = var.project_id
  secret_id = var.secret_name
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run_job_sa.email}"
}