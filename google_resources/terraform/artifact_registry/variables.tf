variable "project_id" {
  type        = string
  description = "Your Google Cloud project ID."
}

variable "region" {
  type        = string
  description = "The GCP region (e.g., 'us-central1')."
}

variable "repository_id" {
  type        = string
  description = "The name for your Artifact Registry Docker repository."
  default     = "fpl-2025"
}

variable "api_services_dependency" {
  description = "Dependency on API services being enabled"
  type        = any
  default     = null
}

variable "github_actions_service_account_email" {
  type        = string
  description = "Email of the GitHub Actions service account for repository access"
}