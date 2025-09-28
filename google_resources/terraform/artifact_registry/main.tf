# Artifact Registry and related resources

# Create an Artifact Registry repository to store your Docker images.
resource "google_artifact_registry_repository" "fpl_2025_repo" {
  project       = var.project_id
  location      = var.region
  repository_id = var.repository_id
  description   = "Docker repository for fpl_2025 jobs"
  format        = "DOCKER"
  depends_on = [
    var.api_services_dependency
  ]
}

# Grant GitHub Actions service account permissions to Artifact Registry
resource "google_artifact_registry_repository_iam_binding" "github_actions" {
  project    = var.project_id
  location   = google_artifact_registry_repository.fpl_2025_repo.location
  repository = google_artifact_registry_repository.fpl_2025_repo.name
  role       = "roles/artifactregistry.writer"
  members    = ["serviceAccount:${var.github_actions_service_account_email}"]
}

resource "google_artifact_registry_repository_iam_binding" "github_actions_reader" {
  project    = var.project_id
  location   = google_artifact_registry_repository.fpl_2025_repo.location
  repository = google_artifact_registry_repository.fpl_2025_repo.name
  role       = "roles/artifactregistry.reader"
  members    = ["serviceAccount:${var.github_actions_service_account_email}"]
}
