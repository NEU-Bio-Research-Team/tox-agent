import type { ReactNode } from 'react';
import { WorkspaceLayout } from './WorkspaceLayout';
import { WorkspaceHeader } from './WorkspaceHeader';
import { cn } from '../../shared/ui/utils';

/** The content widths a workspace page may choose (one per route, recorded here
 * so pages stop inventing their own `max-w-*`). */
const WIDTHS = {
  prose: 'max-w-3xl',
  page: 'max-w-5xl',
  wide: 'max-w-[72rem]',
} as const;

/**
 * A page inside the workspace frame: the slim header with the page title, then a
 * scrolling column of the chosen width. The session workbench does not use this
 * (it manages its own columns) but shares the frame.
 */
export function WorkspacePage({
  title,
  width = 'page',
  headerActions,
  children,
}: {
  title: string;
  width?: keyof typeof WIDTHS;
  headerActions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <WorkspaceLayout>
      <div className="flex h-full min-h-0 flex-col bg-background">
        <WorkspaceHeader title={title} actions={headerActions} breadcrumb />
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className={cn('mx-auto flex w-full flex-col gap-6 px-4 py-8 md:px-8', WIDTHS[width])}>{children}</div>
        </div>
      </div>
    </WorkspaceLayout>
  );
}
