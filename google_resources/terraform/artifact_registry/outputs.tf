# Outputs for artifact registry and GitHub Actions integration

output "repository_name" {
  value       = google_artifact_registry_repository.fpl_2025_repo.name
  description = "The name of the Artifact Registry repository."
}

output "repository_location" {
  value       = google_artifact_registry_repository.fpl_2025_repo.location
  description = "The location of the Artifact Registry repository."
}

output "repository_url" {
  value       = "${google_artifact_registry_repository.fpl_2025_repo.location}-docker.pkg.dev/${google_artifact_registry_repository.fpl_2025_repo.project}/${google_artifact_registry_repository.fpl_2025_repo.repository_id}"
  description = "The full URL of the Artifact Registry repository."
}
