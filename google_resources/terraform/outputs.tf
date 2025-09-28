# This file defines outputs that are useful for configuring your GitHub Actions workflow.

output "workload_identity_provider" {
  value       = module.iam_service.workload_identity_provider
  description = "The full name of the Workload Identity Provider for GitHub Actions."
}

output "github_actions_service_account_email" {
  value       = module.iam_service.github_actions_service_account_email
  description = "The email of the service account for GitHub Actions to impersonate."
}
