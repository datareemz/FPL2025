# IAM Service resources for GitHub Actions integration

# Create a Workload Identity Pool for GitHub Actions
resource "google_iam_workload_identity_pool" "github_pool" {
  project                   = var.project_id
  workload_identity_pool_id = "github-actions-pool"
  display_name              = "GitHub Actions Pool"
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