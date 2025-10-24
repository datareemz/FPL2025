# Outputs for IAM service resources

output "workload_identity_provider" {
  value       = google_iam_workload_identity_pool_provider.github_provider.name
  description = "The full name of the Workload Identity Provider for GitHub Actions."
}

output "github_actions_service_account_email" {
  value       = google_service_account.github_actions_writer.email
  description = "The email of the service account for GitHub Actions to impersonate."
}

output "github_actions_service_account_name" {
  value       = google_service_account.github_actions_writer.name
  description = "The name of the service account for GitHub Actions to impersonate."
}

output "cloud_run_job_service_account_email" {
  value       = google_service_account.cloud_run_job_sa.email
  description = "The email of the service account for Cloud Run job."
}

output "cloud_run_job_service_account_name" {
  value       = google_service_account.cloud_run_job_sa.name
  description = "The name of the service account for Cloud Run job."
}