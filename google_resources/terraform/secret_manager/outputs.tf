output "secret_id" {
  value       = google_secret_manager_secret.fpl_credentials.secret_id
  description = "The ID of the created secret."
}

output "secret_name" {
  value       = google_secret_manager_secret.fpl_credentials.name
  description = "The full resource name of the secret."

}

output "secret_version_name" {
  value       = google_secret_manager_secret_version.fpl_credentials_version.name
  description = "The full resource name of the secret version."
}