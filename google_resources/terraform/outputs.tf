# This file defines outputs that are useful for configuring your GitHub Actions workflow.

output "workload_identity_provider" {
  value       = module.iam_service.workload_identity_provider
  description = "The full name of the Workload Identity Provider for GitHub Actions."
}

output "github_actions_service_account_email" {
  value       = module.iam_service.github_actions_service_account_email
  description = "The email of the service account for GitHub Actions to impersonate."
}

output "cloud_run_job_service_account_email" {
  value       = module.iam_service.cloud_run_job_service_account_email
  description = "The email of the service account used by Cloud Run job."
}

output "bucket_name" {
  value       = module.cloud_storage.bucket_name
  description = "The name of the Cloud Storage bucket."
}

output "bucket_url" {
  value       = module.cloud_storage.bucket_url
  description = "The URL of the Cloud Storage bucket."
}

output "secret_id" {
  value       = module.secret_manager.secret_id
  description = "The ID of the created secret."
}

output "secret_name" {
  value       = module.secret_manager.secret_name
  description = "The full resource name of the secret."
}
