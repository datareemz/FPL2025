# This file defines outputs that are useful for configuring your GitHub Actions workflow.

output "workload_identity_provider" {
  value       = google_iam_workload_identity_pool_provider.github_provider.name
  description = "The full name of the Workload Identity Provider for GitHub Actions."
}

output "github_actions_service_account_email" {
  value       = google_service_account.github_actions_writer.email
  description = "The email of the service account for GitHub Actions to impersonate."
}
